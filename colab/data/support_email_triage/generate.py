"""Deterministic synthetic customer-support email triage data for Unsloth decision models.

Writes train.jsonl / test.jsonl in the typed-decisions row format that
FastDecisionModel.build_dataset reads: every row has `state`, `questions` and
`gold`, each stored as a JSON string (the same layout as
LocalLLaMA/typed-decisions on Hugging Face). Flat `category` / `priority` /
`ambiguous` columns are kept next to them for reporting; they are not model input.

Standard library only, so it runs anywhere:

    python generate.py            # rewrite train.jsonl, test.jsonl, manifest.json
    python generate.py --check    # regenerate in memory and compare with the files on disk
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

SEED = 3407
N_PER_CATEGORY = 250
TEST_FRACTION = 0.2
AMBIGUOUS_RATE = 0.15

OUT_DIR = Path(__file__).resolve().parent

CATEGORIES = {
    "billing": "Charges, invoices, payment methods, plan prices or a charge that looks wrong, when the customer is not asking for money back.",
    "technical_issue": "The product, app, device or website is broken, erroring, slow or not behaving as expected.",
    "account_access": "Logging in, passwords, two-factor codes, locked or compromised accounts, or changing the login email.",
    "refund_request": "The customer explicitly asks to get money back or a credit for something they paid for.",
    "shipping_delivery": "Where an order is, late, lost or damaged deliveries, or changing the address of a shipment.",
    "feature_request": "Suggestions, ideas or requests for new features, integrations or improvements.",
    "cancellation": "The customer wants to cancel a subscription or order, stop a renewal, or close the account.",
    "spam_or_other": "Marketing, sales pitches, phishing, auto-replies, wrong recipient, or anything that is not a support request.",
}

PRIORITY_LEVELS = [
    "Low: no time pressure, such as feedback, ideas, general questions or nothing to act on.",
    "Normal: a real problem for this customer that should be handled in the usual queue.",
    "Urgent: business blocked, money or security at risk, many users affected, or a same-day deadline.",
]

QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "Which support queue should this customer email be routed to?",
        "criteria": CATEGORIES,
    },
    "priority": {
        "type": "score",
        "instructions": "How urgently does support need to respond to this email?",
        "criteria": PRIORITY_LEVELS,
    },
}

# P(priority = 0, 1, 2) per category.
PRIORITY_WEIGHTS = {
    "billing": (0.15, 0.55, 0.30),
    "technical_issue": (0.10, 0.50, 0.40),
    "account_access": (0.05, 0.45, 0.50),
    "refund_request": (0.15, 0.65, 0.20),
    "shipping_delivery": (0.15, 0.55, 0.30),
    "feature_request": (0.80, 0.20, 0.00),
    "cancellation": (0.30, 0.55, 0.15),
    "spam_or_other": (1.00, 0.00, 0.00),
}

# (primary, secondary): the email mentions the secondary topic, but the ask is the primary one.
AMBIGUOUS_PAIRS = {
    "billing": ["refund_request", "account_access", "cancellation"],
    "technical_issue": ["account_access", "feature_request", "refund_request"],
    "account_access": ["technical_issue", "billing"],
    "refund_request": ["billing", "shipping_delivery", "cancellation", "technical_issue"],
    "shipping_delivery": ["refund_request", "technical_issue"],
    "feature_request": ["technical_issue", "cancellation"],
    "cancellation": ["billing", "refund_request", "technical_issue"],
    "spam_or_other": [],
}

# ---------------------------------------------------------------------------
# Slot values. The company is "Lumio": smart-home devices plus a Lumio+ app subscription.
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Alice", "Ben", "Carla", "Dmitri", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jamal",
    "Kate", "Luis", "Mei", "Nikhil", "Olga", "Pedro", "Quinn", "Rosa", "Sam", "Tariq",
    "Uma", "Victor", "Wen", "Xavier", "Yara", "Zoe", "Aaron", "Bianca", "Chen", "Divya",
    "Emeka", "Fatima", "Gustavo", "Hannah", "Ivan", "Julia", "Kofi", "Leila", "Marco", "Nora",
]
LAST_NAMES = [
    "Smith", "Nguyen", "Garcia", "Kowalski", "Okafor", "Tanaka", "Muller", "Rossi", "Silva",
    "Haddad", "Johansson", "Patel", "Kim", "Dubois", "O'Brien", "Novak", "Reyes", "Cohen",
    "Andersen", "Mensah", "Ivanova", "Lopez", "Schmidt", "Wright", "Bauer", "Costa",
]
EMAIL_DOMAINS = ["gmail.com", "outlook.com", "yahoo.com", "icloud.com", "proton.me", "fastmail.com"]
COMPANY_DOMAINS = ["brightdesk.io", "harbor-dental.com", "kitecafe.co", "oakandpine.org", "nimbusretail.com"]
DEVICES = ["Lumio Hub", "Lumio Cam 2", "Lumio Doorbell Pro", "Lumio Smart Plug", "Lumio Thermostat", "Lumio Motion Sensor"]
PLANS = ["Lumio+ Basic", "Lumio+ Plus", "Lumio+ Pro", "Lumio Business"]
PLATFORMS = ["the iOS app", "the Android app", "the web dashboard", "the Mac app", "the Windows app"]
CARRIERS = ["UPS", "FedEx", "DHL", "USPS", "Royal Mail", "Canada Post"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ERRORS = ["Error 503", "\"Something went wrong (code 1042)\"", "a blank white screen", "\"Device offline\"",
          "\"Sync failed: timeout\"", "a spinning wheel that never stops", "\"Unexpected token\" in a red box"]
INTEGRATIONS = ["Google Home", "Alexa", "Apple HomeKit", "IFTTT", "Home Assistant", "Slack", "Zapier", "SmartThings"]


def _money(rng):
    return f"${rng.choice([4.99, 9.99, 14.99, 19.99, 29.99, 49.00, 89.00, 129.00, 199.00, 249.99]):.2f}"


def _slots(rng):
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    business = rng.random() < 0.3
    domain = rng.choice(COMPANY_DOMAINS) if business else rng.choice(EMAIL_DOMAINS)
    handle = rng.choice([f"{first}.{last}", f"{first[0]}{last}", f"{first}{rng.randint(1, 99)}"])
    return {
        "first": first,
        "last": last,
        "sender": f"{handle.lower().replace(chr(39), '')}@{domain}",
        "business": business,
        "device": rng.choice(DEVICES),
        "device2": rng.choice(DEVICES),
        "plan": rng.choice(PLANS),
        "plan2": rng.choice(PLANS),
        "platform": rng.choice(PLATFORMS),
        "carrier": rng.choice(CARRIERS),
        "order": f"LM-{rng.randint(10000, 99999)}",
        "invoice": f"INV-2026-{rng.randint(1000, 9999)}",
        "tracking": f"1Z{rng.randint(100000000, 999999999)}",
        "amount": _money(rng),
        "amount2": _money(rng),
        "month": rng.choice(MONTHS),
        "day": rng.choice(WEEKDAYS),
        "days": rng.choice([3, 4, 5, 6, 8, 10, 12, 14]),
        "n": rng.choice([2, 3, 4, 5, 12, 30, 40]),
        "seats": rng.choice([5, 8, 12, 25, 40]),
        "error": rng.choice(ERRORS),
        "integration": rng.choice(INTEGRATIONS),
        "last4": f"{rng.randint(1000, 9999)}",
        "city": rng.choice(["Austin", "Leeds", "Toronto", "Berlin", "Manila", "Lyon", "Denver", "Melbourne"]),
    }


# ---------------------------------------------------------------------------
# Category content. Each ask is (subjects that fit this ask, ask text). The subject
# is either one of those or a generic subject for the category. `details` add
# supporting context, and `mention` is used when another category's email brushes
# against this topic (the ambiguous rows).
# ---------------------------------------------------------------------------

CONTENT = {
    "billing": {
        "generic_subjects": ["Billing question", "Question about my bill", "Billing issue", "Account charges", "Invoice question"],
        "asks": [
            (["Charged twice?", "Double charge"], "I was charged {amount} twice this month for {plan} and I only have one subscription. Can you check what happened?"),
            (["Invoice {invoice} amount", "Wrong amount on my bill"], "My invoice {invoice} shows {amount2} but my plan is supposed to be {amount} a month."),
            (["Charge I don't recognize", "Unknown charge from Lumio"], "There's a charge from Lumio for {amount} on my card ending {last4} that I don't recognize. What is it for?"),
            (["Why did my price go up", "Price increase?"], "Can you explain why my {plan} renewal went up from {amount} to {amount2}?"),
            (["Update payment method", "New card"], "I need to update the card on file, the old one ending {last4} expired and I can't find where to change it."),
            (["Need a VAT invoice", "Invoice with company details"], "Could you send me an invoice with our company VAT number on it for the {month} payment?"),
            (["Payment failed", "Payment error but money taken"], "Your system says my payment failed but my bank shows the money left my account."),
            (["Prorated charge", "Upgrade charge looks wrong"], "I upgraded to {plan2} mid-month and the prorated charge looks wrong to me."),
            (["Seat count on invoice", "Billed for too many seats"], "We were billed for {seats} seats on {plan} but we only use 3. Please correct the invoice."),
            (["Switch to annual billing", "Annual plan"], "How do I switch from monthly to annual billing on {plan}? I'd like the discount."),
            (["Overdue payment notice", "Dunning email"], "I got an email saying my payment is overdue, but autopay is on and the card is valid."),
            (["Receipt missing tax", "Receipt for {invoice}"], "The receipt for invoice {invoice} is missing the tax line our accountant needs."),
        ],
        "details": [
            "The charges were on {month} 3 and {month} 4.",
            "My card ends in {last4}.",
            "I've attached a screenshot of my bank statement.",
            "I checked the billing page in {platform} and it only lists one plan.",
            "Our finance team flagged it during month-end close.",
            "The account is under {sender}.",
            "This has happened two months in a row now.",
        ],
        "mention": [
            "I also noticed an odd charge on my last statement.",
            "Separately, my last invoice looked a bit higher than usual.",
        ],
    },
    "technical_issue": {
        "generic_subjects": ["App not working", "Something is broken", "Problem with my device", "Help - not working", "Bug report"],
        "asks": [
            (["{device} keeps going offline", "Device offline again"], "My {device} keeps going offline every few hours and I have to unplug it to get it back."),
            (["Error when I open the app", "Bug in {platform}"], "Since the last update {platform} shows {error} as soon as I open it."),
            (["Camera feed is black", "No live view"], "The live view on my {device} is just black, but the recordings still save."),
            (["Firmware update failed", "Stuck firmware update"], "The firmware update on my {device} got stuck at 80% and now the light just blinks red."),
            (["Automations stopped running", "Schedules not working"], "My automations stopped triggering on {day}, nothing runs at the scheduled times anymore."),
            (["Notifications delayed", "Late alerts"], "Notifications from the {device} arrive 10 to 15 minutes late, which makes them useless."),
            (["Can't pair new device", "Pairing error"], "When I try to add a second {device2} the pairing fails with {error}."),
            (["App crashes", "{platform} crashing"], "{platform} crashes whenever I open the history tab."),
            (["Wrong temperature reading", "Thermostat reading off"], "The {device} reports the wrong temperature, about 6 degrees off from the room."),
            (["All devices offline at office", "Devices won't reconnect"], "Our {n} {device}s at the office all dropped off the network after the router reboot and won't reconnect."),
            (["{integration} integration broken", "{integration} not working"], "The {integration} integration stopped working, it says the device is unreachable."),
            (["Can't download clips", "Clip download fails"], "Video clips won't download, the progress bar gets to the end and then nothing happens."),
        ],
        "details": [
            "I've already tried restarting the hub and reinstalling the app.",
            "I'm on the latest version of {platform}.",
            "My Wi-Fi is fine, every other device in the house works.",
            "It started right after the update on {day}.",
            "I factory reset it twice, same result.",
            "Logs are attached if that helps.",
            "Device serial is SN-{last4}{last4}.",
        ],
        "mention": [
            "The app has also been a bit glitchy lately.",
            "On top of that, the {device} has been dropping offline now and then.",
        ],
    },
    "account_access": {
        "generic_subjects": ["Account help", "Help accessing account", "Login problem", "Account issue"],
        "asks": [
            (["Can't log in", "Password not accepted"], "I can't log in, it keeps saying my password is wrong even after I reset it."),
            (["Password reset not working", "No reset email"], "The password reset email never arrives, I've checked spam and tried three times."),
            (["2FA code never arrives", "Verification code problem"], "My two-factor codes stopped arriving by SMS so I'm stuck at the verification screen."),
            (["Someone accessed my account", "Suspicious login"], "I got a login alert from {city} and it wasn't me, I think someone got into my account."),
            (["Change login email", "Update account email"], "I need to change the email I log in with from my old work address to {sender}."),
            (["Account locked", "Locked out of my account"], "My account got locked after too many attempts and the unlock link has expired."),
            (["Lost my phone with authenticator", "Lost 2FA device"], "I lost my phone and the authenticator app with it, how do I get back into my account?"),
            (["Login loop", "Keeps sending me back to login"], "After I log in on {platform} it sends me straight back to the login page."),
            (["Transfer admin access", "Admin left the company"], "Our admin left the company and we need the owner role on our account moved to me."),
            (["Account not found", "It says I have no account"], "It says no account exists for {sender}, but I've been a customer for two years."),
        ],
        "details": [
            "The account email is {sender}.",
            "I can still see the cameras on my partner's phone, so the devices are fine.",
            "I'm happy to verify my identity with my last invoice number, {invoice}.",
            "I tried both {platform} and a different browser.",
            "My phone number changed last month, maybe that's why.",
        ],
        "mention": [
            "I also had trouble logging in earlier this week, but it eventually worked.",
            "(I had to reset my password just to get in and write this.)",
        ],
    },
    "refund_request": {
        "generic_subjects": ["Refund please", "Requesting a refund", "I want my money back", "Refund request"],
        "asks": [
            (["Refund after downgrade", "Charged after downgrading"], "Please refund the {amount} I was charged for {plan} after I already downgraded."),
            (["Refund for order {order}", "Return and refund"], "I'd like a full refund for order {order}, the {device} isn't what I expected."),
            (["Refund for duplicate charge", "Refund request - {invoice}"], "I was billed twice for invoice {invoice}, please refund the duplicate {amount}."),
            (["Broken on arrival - refund", "Money back for order {order}"], "The {device} arrived broken and I'd rather have my money back than a replacement."),
            (["Trial charge refund", "Forgot to cancel trial"], "I forgot to cancel the free trial and got charged {amount2}. Can I get a refund since I never used it?"),
            (["Credit for outage", "Compensation for downtime"], "Your service was down for {days} days, I'd like a credit or partial refund for that time."),
            (["Refund not received", "Still waiting on refund"], "I returned the {device} {days} days ago and still haven't seen the refund on my card."),
            (["Unauthorized renewal refund", "Refund renewal charge"], "Please put the {amount} back on my card ending {last4}, I didn't authorize that renewal."),
            (["Bought wrong plan", "Refund the difference"], "I bought {plan2} by mistake instead of {plan}, can you refund the difference?"),
        ],
        "details": [
            "Order number is {order}.",
            "The return was dropped off with {carrier}, tracking {tracking}.",
            "I've been a customer for years and never asked for this before.",
            "Your policy page says 30 days and I'm well within that.",
            "The charge is on my card ending {last4}.",
        ],
        "mention": [
            "Honestly I'd like some money back for the trouble, but mostly I just want this fixed.",
            "If this can't be sorted I'll probably ask for a refund.",
        ],
    },
    "shipping_delivery": {
        "generic_subjects": ["Delivery question", "My order", "Shipping issue", "Order status"],
        "asks": [
            (["Where is my order?", "Order {order} not arrived"], "My order {order} was supposed to arrive {days} days ago and tracking hasn't moved."),
            (["Package marked delivered but not here", "Missing package"], "{carrier} says the package was delivered but there's nothing at my door or with the neighbours."),
            (["Damaged in shipping", "Box arrived crushed"], "The box for order {order} arrived crushed and the {device} inside is cracked. Can you send a replacement?"),
            (["Change delivery address", "Moving - new address"], "Can I change the delivery address for order {order}? I'm moving to {city} next week."),
            (["Tracking hasn't updated", "Tracking stuck"], "Tracking number {tracking} just says label created, it's been a week."),
            (["Wrong item received", "Received the wrong device"], "I ordered a {device} but received a {device2} instead."),
            (["Shipping to {city}", "Do you deliver here?"], "Do you ship to {city}? The checkout page won't accept my postcode."),
            (["Missing item from order", "Only half my order arrived"], "Only one of the two items in order {order} showed up."),
            (["Faster shipping?", "Need it by {day}"], "I need the {device} delivered before {day}, can you upgrade the shipping?"),
        ],
        "details": [
            "I ordered on {month} 2 with standard shipping.",
            "The carrier is {carrier}, tracking {tracking}.",
            "Photos of the box are attached.",
            "I've checked with the building's front desk.",
            "It was a gift, so the timing matters.",
        ],
        "mention": [
            "The order took ages to arrive too, but it got here eventually.",
            "(The {device} also arrived a week later than promised.)",
        ],
    },
    "feature_request": {
        "generic_subjects": ["Feature suggestion", "Idea for the app", "Suggestion", "Feedback", "Any plans for this?"],
        "asks": [
            (["Please add {integration} support", "{integration} integration?"], "It would be great if the {device} worked with {integration}."),
            (["Would love dark mode", "Dark mode request"], "Any chance of a dark mode in {platform}? It's really bright at night."),
            (["Request: bulk export clips", "Export a whole day"], "Could you add a way to export a whole day of camera clips at once?"),
            (["Family sharing", "Shared access idea"], "I'd love to be able to share access with family without giving them my password."),
            (["Recording schedule idea", "Away-only recording"], "Please add a schedule option so the {device} only records when nobody is home."),
            (["Battery overview page", "Dashboard suggestion"], "It would help a lot if the web dashboard showed battery levels for all devices on one page."),
            (["Home screen widget", "Widget for {platform}"], "Have you thought about a widget for the home screen on {platform}?"),
            (["API / webhooks", "Developer API?"], "Can you add an API or webhooks so we can plug Lumio into our own tools?"),
            (["Vacation mode idea", "Feature idea: vacation mode"], "A vacation mode that turns lights on and off randomly would be really useful."),
            (["Custom alert sounds", "Suggestion: alert sounds"], "Would you consider letting us rename the alert sounds or upload our own?"),
        ],
        "details": [
            "Other than that I really like the product.",
            "A few friends with Lumio setups asked me about this too.",
            "Not urgent at all, just an idea.",
            "We'd probably upgrade to {plan2} if this existed.",
            "Happy to beta test it.",
        ],
        "mention": [
            "Also, it would be nice if the app let me choose the notification sound.",
            "Side note: a dark mode would be lovely some day.",
        ],
    },
    "cancellation": {
        "generic_subjects": ["Cancellation", "Please cancel", "Cancel my account", "Cancel request"],
        "asks": [
            (["Cancel my subscription", "Cancel {plan}"], "Please cancel my {plan} subscription, I don't need it anymore."),
            (["Close my account", "Delete my account"], "I want to close my Lumio account and delete my data."),
            (["Stop auto-renewal", "Turn off renewal"], "Can you turn off auto-renew? I don't want to be charged again next month."),
            (["Cancel order {order}", "Cancel my order"], "Please cancel order {order} before it ships."),
            (["Ending our contract", "Cancel all seats"], "We're moving to a different provider and need to cancel our {plan} plan for all {seats} seats."),
            (["How do I cancel?", "Can't find cancel button"], "How do I cancel? There's no button for it in {platform}."),
            (["End subscription - moving abroad", "Cancel at end of period"], "I'm moving abroad, so please end my subscription at the end of this billing period."),
            (["Remove add-on", "Cancel {plan} add-on"], "Cancel the {plan} add-on but keep the basic account please."),
        ],
        "details": [
            "The account is {sender}.",
            "No hard feelings, it just isn't for us.",
            "Please confirm by email once it's done.",
            "The renewal date is {month} 28.",
            "We'll keep the devices but won't need the cloud plan.",
        ],
        "mention": [
            "If this keeps happening I'm going to cancel.",
            "I've been thinking about cancelling, to be honest.",
        ],
    },
}

# (subjects, body, sender kind). "external" senders use a bulk-mail domain.
SPAM = [
    (["Boost your SEO rankings today", "Page one of Google in 30 days"], "We help companies like Lumio rank on page one of Google in 30 days. Can I send over a free audit?", "external"),
    (["Partnership opportunity", "Influencer campaign for Lumio"], "I run a growth agency and we'd love to partner with you on influencer campaigns. Are you the right person to talk to?", "external"),
    (["Out of Office", "Automatic reply: {first} {last}"], "I am out of the office until {day} with limited access to email. For urgent matters please contact my colleague.", "person"),
    (["Automatic reply", "Re: your message"], "Thank you for your email. This is an automatic reply to confirm we received your message.", "external"),
    (["Your mailbox will be suspended", "Action required: verify account"], "Dear user, your mailbox storage is full. Verify your account at the link below within 24 hours or it will be suspended.", "phishing"),
    (["Re: invoice", "Invoice attached"], "Please find the attached invoice for payment. Open the document to view the amount due.", "phishing"),
    (["Thanks!", "All sorted"], "Just wanted to say the issue I emailed about yesterday sorted itself out. No need to reply, thanks!", "person"),
    (["Congratulations!", "You've been selected"], "Congratulations, you have been selected for a $500 gift card. Click here to claim your reward.", "phishing"),
    (["Job application", "Application: support specialist"], "I'm applying for the customer support role I saw on LinkedIn. My CV is attached.", "person"),
    (["Wrong email?", "Sorry, wrong address"], "I think I sent this to the wrong address, sorry. Please ignore.", "person"),
    (["Guest post offer", "Cheap backlinks"], "We offer guest posting on high authority blogs, 50 USD per article. Interested?", "external"),
    (["Unsubscribe", "Stop emailing me"], "Please remove me from your mailing list.", "person"),
    (["Wholesale office furniture", "Vendor introduction"], "Are you the person who handles vendor onboarding? We sell office furniture at wholesale prices.", "external"),
    (["test", "Testing"], "Just testing whether this address works. Please ignore.", "person"),
    (["Love my Lumio!", "Just a thank you"], "No question, just wanted to say the doorbell caught a package thief last week and the police got him. Great product!", "person"),
]

# ---------------------------------------------------------------------------
# Priority cues and tone wrappers.
# ---------------------------------------------------------------------------

PRIORITY_CUES = {
    0: [
        "No rush on this.", "Whenever you get a chance.", "Not urgent, just curious.",
        "Take your time.", "", "", "",
    ],
    1: [
        "Could you take a look?", "Thanks in advance for looking into it.", "Let me know what you find.",
        "Hoping to get this sorted this week.", "", "", "",
    ],
    2: [
        "This is urgent, I need it fixed today.",
        "Please treat this as an emergency.",
        "I need this resolved before {day} morning or we lose a client.",
        "I need an answer today please.",
    ],
}

URGENT_CUES_BY_CATEGORY = {
    "billing": ["My card is maxed out because of this, please act today.", "Our accounts payable run closes today and this has to be right."],
    "technical_issue": ["Our whole office is affected and we can't work.", "This is our front door camera and it's our only security, please hurry.",
                        "We have {n} locations down at the moment."],
    "account_access": ["I'm worried someone is using my account right now.", "I can't see my cameras while I'm away and I need to check on my house today."],
    "refund_request": ["My card is maxed out because of this, please act today.", "I need that money back before my rent is due this week."],
    "shipping_delivery": ["It's a birthday gift for {day} and I can't wait any longer.", "We open a new store on {day} and need these installed before then."],
    "cancellation": ["It renews tomorrow, so please cancel it today.", "Please stop it before the next charge goes out tonight."],
}

PRIORITY_SUBJECT_PREFIX = {0: [""], 1: ["", "", "Re: "], 2: ["URGENT: ", "Urgent - ", "[ASAP] ", "", ""]}

TONES = {
    "polite": {
        "greet": ["Hi there,", "Hello,", "Hi Lumio team,", "Good morning,", "Dear Support,"],
        "close": ["Thanks so much!", "Many thanks,", "Thank you for your help.", "Kind regards,", "Best,"],
        "extra": ["I really appreciate it.", "Hope you're having a good day."],
    },
    "neutral": {
        "greet": ["Hi,", "Hello,", "Hi support,", ""],
        "close": ["Thanks,", "Regards,", "Cheers,", ""],
        "extra": [""],
    },
    "frustrated": {
        "greet": ["Hi,", "Hello again,", ""],
        "close": ["Please sort this out.", "Thanks.", "Waiting for a reply.", ""],
        "extra": ["This is the second time I'm writing about this.", "Honestly this is getting really frustrating.",
                  "I've been waiting for days with no answer."],
    },
    "angry": {
        "greet": ["", "To whoever reads this,", "Hello?"],
        "close": ["Fix it.", "I expect a reply TODAY.", "Unacceptable.", ""],
        "extra": ["This is ridiculous.", "Worst customer experience I've had in years.", "I'm about to post about this online."],
    },
    "terse": {
        "greet": [""],
        "close": ["", "thx", "ty"],
        "extra": [""],
    },
    "rambling": {
        "greet": ["Hi there, hope this is the right place to ask,", "Hello! Long time customer here,"],
        "close": ["Anyway, thanks for reading all this!", "Sorry for the long email, thanks!"],
        "extra": ["We got our first Lumio setup when we moved into this house a couple of years ago and it's been mostly great.",
                  "My kids are the ones who usually deal with the tech, but they're away at school.",
                  "I'm not very techy so bear with me.",
                  "I meant to write about this a while ago but life got busy."],
    },
}
TONE_WEIGHTS = {"polite": 3, "neutral": 3, "frustrated": 2, "angry": 1, "terse": 1.5, "rambling": 1.5}


def _fill(rng, template, slots):
    return template.format(**slots)


def _pick_weighted(rng, weights):
    keys = list(weights)
    return rng.choices(keys, weights = [weights[k] for k in keys], k = 1)[0]


def _signature(rng, slots, tone):
    if tone == "terse" or rng.random() < 0.2:
        return ""
    name = rng.choice([slots["first"], f"{slots['first']} {slots['last']}", slots["first"][0]])
    if slots["business"] and rng.random() < 0.6:
        return f"{name}\nOperations, {slots['sender'].split('@')[1].split('.')[0].replace('-', ' ').title()}"
    return name


def _lowercase(text):
    return text[:1].lower() + text[1:] if text else text


def _support_email(rng, category, priority, secondary):
    content = CONTENT[category]
    slots = _slots(rng)
    tone = _pick_weighted(rng, TONE_WEIGHTS)
    if priority == 2 and tone == "rambling":
        tone = "frustrated"
    style = TONES[tone]

    length = rng.choices(["short", "medium", "long"], weights = [3, 5, 2], k = 1)[0]
    if tone == "terse":
        length = "short"

    greeting = rng.choice(style["greet"]) if length != "short" or rng.random() < 0.3 else ""
    sentences = []
    if tone == "rambling" or (length == "long" and rng.random() < 0.6):
        sentences.append(rng.choice(style["extra"]))
    mention = _fill(rng, rng.choice(CONTENT[secondary]["mention"]), slots) if secondary else ""
    mention_first = rng.random() < 0.5
    if mention and mention_first:
        sentences.append(mention)
    ask_subjects, ask = rng.choice(content["asks"])
    sentences.append(_fill(rng, ask, slots))
    n_details = {"short": rng.choice([0, 0, 1]), "medium": rng.choice([1, 1, 2]), "long": rng.choice([2, 3])}[length]
    for detail in rng.sample(content["details"], k = min(n_details, len(content["details"]))):
        sentences.append(_fill(rng, detail, slots))
    if mention and not mention_first:
        sentences.append(mention)
    if tone in ("frustrated", "angry"):
        sentences.append(rng.choice(style["extra"]))
    cues = PRIORITY_CUES[priority] + (URGENT_CUES_BY_CATEGORY.get(category, []) * 2 if priority == 2 else [])
    sentences.append(_fill(rng, rng.choice(cues), slots))
    sentences = [s for s in sentences if s]
    closing = rng.choice(style["close"])
    signature = _signature(rng, slots, tone)

    if tone == "terse":
        body = ". ".join(_lowercase(s).rstrip(".") for s in sentences + [closing] if s)
    else:
        if len(sentences) > 2 and rng.random() < 0.5:
            cut = rng.randint(1, len(sentences) - 1)
            middle = " ".join(sentences[:cut]) + "\n\n" + " ".join(sentences[cut:])
        else:
            middle = " ".join(sentences)
        sign_off = "\n".join(x for x in [closing, signature] if x)
        body = "\n\n".join(x for x in [greeting, middle, sign_off] if x)

    if tone == "angry" and rng.random() < 0.4:
        body = body.replace(".", "!", 1)

    subject_pool = ask_subjects if rng.random() < 0.65 else content["generic_subjects"]
    subject = rng.choice(PRIORITY_SUBJECT_PREFIX[priority]) + _fill(rng, rng.choice(subject_pool), slots)
    if rng.random() < 0.06:
        subject = rng.choice(["Help", "Question", "(no subject)", "hi", "Support"])
    if tone == "terse" and rng.random() < 0.5:
        subject = subject.lower()
    return {"from": slots["sender"], "subject": subject, "body": body}, tone, length


def _spam_email(rng):
    slots = _slots(rng)
    subjects, body, sender_kind = rng.choice(SPAM)
    body = _fill(rng, body, slots)
    if sender_kind == "person":
        sender = slots["sender"]
        sign = rng.choice([slots["first"], f"{slots['first']} {slots['last']}", ""])
    elif sender_kind == "phishing":
        sender = f"{rng.choice(['security', 'billing', 'admin', 'noreply'])}@{rng.choice(['lumio-billing-secure.info', 'account-verify-mail.com', 'mailbox-support.net'])}"
        sign = rng.choice(["IT Support Team", "Account Security", ""])
    else:
        sender = f"{rng.choice(['promo', 'info', 'growth', 'hello', 'partners'])}@{rng.choice(['mail-blast.biz', 'seo-rank.net', 'corp-mailer.com', 'growthly.agency'])}"
        sign = rng.choice([f"{slots['first']} {slots['last']}\nPartnerships", "Best regards,\nMarketing Team", ""])
    greeting = rng.choice(["Hi,", "Hello,", "Dear Sir/Madam,", "Hey there,", "", ""])
    body = "\n\n".join(x for x in [greeting, body, sign] if x)
    subject = _fill(rng, rng.choice(subjects), slots)
    return {"from": sender, "subject": subject, "body": body}, "neutral", "short"


def _gold(category, priority, secondary, rng):
    if secondary is None:
        cat_probs = {category: 1.0}
    else:
        primary = round(rng.uniform(0.6, 0.75), 2)
        cat_probs = {category: primary, secondary: round(1.0 - primary, 2)}
    return {
        "category": {
            "type": "choice",
            "label": category,
            "probabilities": {key: cat_probs.get(key, 0.0) for key in CATEGORIES},
        },
        "priority": {
            "type": "score",
            "label": str(priority),
            "probabilities": {str(level): float(level == priority) for level in range(len(PRIORITY_LEVELS))},
        },
    }


def generate():
    rng = random.Random(SEED)
    rows, seen = [], set()
    for category in CATEGORIES:
        made = 0
        while made < N_PER_CATEGORY:
            priority = rng.choices([0, 1, 2], weights = PRIORITY_WEIGHTS[category], k = 1)[0]
            secondary = None
            if AMBIGUOUS_PAIRS[category] and rng.random() < AMBIGUOUS_RATE:
                secondary = rng.choice(AMBIGUOUS_PAIRS[category])
            if category == "spam_or_other":
                state, tone, length = _spam_email(rng)
            else:
                state, tone, length = _support_email(rng, category, priority, secondary)
            key = (state["subject"], state["body"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "category": category,
                    "priority": priority,
                    "ambiguous": secondary is not None,
                    "secondary_category": secondary,
                    "tone": tone,
                    "length": length,
                    "state": state,
                    "gold": _gold(category, priority, secondary, rng),
                }
            )
            made += 1

    train, test = [], []
    for category in CATEGORIES:
        group = [row for row in rows if row["category"] == category]
        rng.shuffle(group)
        cut = int(round(len(group) * TEST_FRACTION))
        test.extend(group[:cut])
        train.extend(group[cut:])
    rng.shuffle(train)
    rng.shuffle(test)

    def finalize(split_rows, split):
        out = []
        for i, row in enumerate(split_rows):
            out.append(
                {
                    "id": f"{split}_{i:05d}",
                    "split": split,
                    "category": row["category"],
                    "priority": row["priority"],
                    "ambiguous": row["ambiguous"],
                    "secondary_category": row["secondary_category"],
                    "tone": row["tone"],
                    "length": row["length"],
                    "state": json.dumps(row["state"], ensure_ascii = False),
                    "questions": json.dumps(QUESTIONS, ensure_ascii = False),
                    "gold": json.dumps(row["gold"], ensure_ascii = False),
                }
            )
        return out

    return finalize(train, "train"), finalize(test, "test")


def _jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii = False, sort_keys = True) + "\n" for row in rows)


def _manifest(files):
    manifest = {"seed": SEED, "generator": "generate.py", "files": {}}
    for name, (text, rows) in files.items():
        manifest["files"][name] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "category": dict(sorted(Counter(r["category"] for r in rows).items())),
            "priority": dict(sorted(Counter(str(r["priority"]) for r in rows).items())),
            "ambiguous": sum(r["ambiguous"] for r in rows),
        }
    return json.dumps(manifest, indent = 2) + "\n"


def main():
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    parser.add_argument("--check", action = "store_true", help = "verify the files on disk match a fresh run")
    parser.add_argument("--out", type = Path, default = OUT_DIR)
    args = parser.parse_args()

    train, test = generate()
    files = {"train.jsonl": (_jsonl(train), train), "test.jsonl": (_jsonl(test), test)}
    outputs = {name: text for name, (text, _) in files.items()}
    outputs["manifest.json"] = _manifest(files)

    if args.check:
        stale = [name for name, text in outputs.items()
                 if not (args.out / name).is_file() or (args.out / name).read_text(encoding = "utf-8") != text]
        if stale:
            raise SystemExit(f"Out of date: {', '.join(stale)}. Run python generate.py")
        print("Dataset files match the generator.")
        return

    args.out.mkdir(parents = True, exist_ok = True)
    for name, text in outputs.items():
        (args.out / name).write_text(text, encoding = "utf-8")
    print(outputs["manifest.json"])


if __name__ == "__main__":
    main()
