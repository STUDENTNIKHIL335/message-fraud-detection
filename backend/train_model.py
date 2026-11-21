"""
ScamShield – ML Model Trainer
Trains a TF-IDF + Logistic Regression pipeline on realistic Indian & global scam messages vs genuine messages.
Saves model artifact to scam_model.joblib.
"""

import os
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Curated dataset: 1 = Scam / Phishing, 0 = Legitimate / Safe
DATASET = [
    # --- SCAMS: Bank KYC & Account Suspended (1) ---
    ("Dear SBI user, your YONO account has been suspended due to pending KYC. Update PAN immediately at http://sbi-kyc-verify.top or your account will be blocked.", 1),
    ("HDFC Bank Alert: Your netbanking will be locked in 24 hours. Please verify your Aadhaar card details here: http://bit.ly/hdfc-kyc-update", 1),
    ("Dear ICICI customer, your debit card is blocked. Kindly click http://icici-bank-support.xyz/unblock to reactivate immediately.", 1),
    ("URGENT: Axis Bank account #XXXX1209 deactivated. Click to re-submit KYC documents: http://axis-verify.click to prevent permanent closure.", 1),
    ("PNB Alert: Your bank account will be frozen today due to incomplete KYC. Complete online verification at http://pnb-kyc-portal.tk", 1),
    ("Kotak 811: Important notice! Your account is put on hold. Update mobile number and PAN details at http://tinyurl.com/kotak-kyc-online", 1),
    ("Bank Warning: Unauthorized login attempt detected from IP 192.168.1.1. If not you, cancel transaction now: http://bank-secure-auth.xyz/cancel", 1),
    ("Dear customer, your Canara Bank account has been credited with ₹25,000 but put on hold. Click http://canara-fund-release.top to claim.", 1),
    ("BOB Alert: Your Bank of Baroda account access will expire in 2 hours. Update KYC now: http://is.gd/bob_kyc_update", 1),
    ("Dear SBI Customer, your YONO reward points worth ₹9,850 are expiring today. Redeem to your bank account immediately: http://sbi-reward-points.site", 1),

    # --- SCAMS: Electricity & Utility Disconnection (1) ---
    ("Dear consumer, your electricity power will be disconnected tonight at 9:30 PM from the electric office because your previous month bill was not updated. Call electricity officer at 9876543210 immediately.", 1),
    ("URGENT POWER NOTICE: Electricity power will be cut tonight at 10:00 PM. Please contact our electricity support manager at 8765432109 or visit http://wbseb-bill-pay.xyz", 1),
    ("TNEB Electricity alert: Your power supply connection will be disconnected by evening. Clear unpaid bill of Rs 1,450 now on http://tneb-onlinebill.top", 1),
    ("BSES Alert: Your power line disconnected order issued. Call electricity helpline 9123456780 to cancel disconnection immediately.", 1),
    ("Mahanagar Gas: Gas pipeline connection will be terminated tomorrow due to pending verification. Contact officer urgently at 9988776655.", 1),

    # --- SCAMS: Courier & India Post Delivery (1) ---
    ("India Post: Your package #IN89218 cannot be delivered due to incomplete street address. Please update your address within 12 hours: http://indiapost-parcel.top/update", 1),
    ("Speed Post: Delivery failed for parcel CP39102910IN. Please pay customs re-delivery fee of ₹25 immediately at http://bit.ly/speedpost-pay", 1),
    ("FedEx: Your shipment is on hold at Mumbai customs. Pay duty tax of Rs 480 to release shipment: http://fedex-customs-clearance.xyz", 1),
    ("DHL Express: Important notice regarding parcel #49102. Action required to confirm delivery address: http://dhl-address-verify.site/confirm", 1),
    ("BlueDart alert: Your parcel could not be delivered. Reschedule delivery date and pay ₹15 fee: http://bluedart-redelivery.club", 1),
    ("India Post Office: Last reminder! Parcel will be returned to sender today. Click http://tinyurl.com/indiapost-tracking-in to claim.", 1),

    # --- SCAMS: Lottery, KBC & Cash Prizes (1) ---
    ("Congratulations! You have won ₹25,00,000 in Kaun Banega Crorepati (KBC) Lucky Draw 2026. Send your bank details and photo on WhatsApp to +91-9876501234 to claim.", 1),
    ("Dear Jio User, your mobile number won ₹10,00,000 cash prize in Jio 5G anniversary lottery. Contact WhatsApp manager Mr. Rana Pratap at 9988112233.", 1),
    ("Amazon Mega Festive Draw: You are selected for lucky prize of iPhone 16 Pro Max! Pay shipping fee ₹299 to claim: http://amazon-festive-gift.xyz", 1),
    ("CONGRATULATIONS! Your mobile number won 2nd prize in Coca Cola promotional sweepstakes. Claim £500,000 via email: claims@cocacola-winner.top", 1),
    ("You have won $5,000,000 USD from the Bill & Melinda Gates Foundation grant program. Contact claims agent immediately at foundation-grant@claim-usd.xyz", 1),

    # --- SCAMS: Fake Govt Schemes & Subsidies (1) ---
    ("PM Mudra Loan Scheme 2026: Approved loan of ₹5,00,000 with 50% government subsidy and 0% interest. Register today: http://bit.ly/pm-mudra-yojana-apply", 1),
    ("Pradhan Mantri Free Smartphone & Laptop Yojana: All students can receive free 5G smartphone. Apply before 30th Nov: http://pm-free-smartphone.club", 1),
    ("PM Kisan Samman Nidhi: 17th installment of ₹6,000 is pending. Update Aadhaar details urgently to credit amount: http://pmkisan-aadhaar-link.xyz", 1),
    ("Govt of India Solar Rooftop Subsidy: Get 100% free solar panel installation approved under PM Surya Ghar Yojana. Register: http://pm-suryaghar-apply.site", 1),
    ("Mahila Samman Yojana: Free ₹2,500 monthly transfer approved for your family. Submit bank verification at http://mahila-yojana-gov.tk", 1),

    # --- SCAMS: Part-Time Job, Telegram & Crypto Investment (1) ---
    ("Work From Home Opportunity: Earn ₹3,000 - ₹8,000 daily by simply liking YouTube videos & rating Google Maps hotels. Contact HR on WhatsApp: https://wa.me/919988001122", 1),
    ("Amazon HR: We are hiring remote freelancers for part time review jobs. No experience needed, daily payout of ₹5,000. Join Telegram: t.me/amazon_hr_tasks", 1),
    ("Earn 300% profit in 24 hours with AI Crypto Trading bot. Guaranteed returns with zero risk. Register now: http://cryptogain-pro.top and deposit $50.", 1),
    ("Special VIP job offer: Work 1 hour per day and earn Rs 20,000 weekly from your smartphone. Limited slots available, click http://rb.gy/job-vip-hire", 1),
    ("Binance Investment signal: Earn 10x guaranteed returns on Bitcoin arbitrage. Join official private channel: http://is.gd/crypto_signals_vip", 1),

    # --- SCAMS: SIM Card, TRAI, Police Summons (1) ---
    ("TRAI Notice: Your SIM card will be permanently blocked in 2 hours due to illegal message broadcast. Call verification desk 9821098210 to stop.", 1),
    ("Airtel Alert: Your 5G e-SIM upgrade request is received. If you did not request, click immediately to cancel: http://airtel-esim-cancel.site", 1),
    ("Delhi Police Cyber Crime Cell: Arrest warrant issued under your name for money laundering. Join urgent video interrogation: http://cybercrime-notice.xyz/court", 1),
    ("Income Tax Department: Outstanding tax refund of ₹18,450 approved for PAN XXXX7890. Confirm bank account details at http://incometax-refund-in.top", 1),
    ("Credit Card Department: Your HDFC credit card points worth Rs 7,500 expire at midnight. Convert into cash directly: http://creditcard-points-cash.click", 1),

    # --- SCAMS: Additional Global & Indian Phishing (1) ---
    ("Netflix Payment declined: Your subscription is on hold. Update credit card billing info within 24h: http://netflix-billing-update.xyz", 1),
    ("Apple Security Alert: Your Apple ID was used to purchase iPhone on Apple Store Tokyo. If this wasn't you, lock ID immediately: http://appleid-security-lock.top", 1),
    ("PayPal Alert: Unusual account activity detected. Your funds are locked. Verify identity at http://paypal-resolution-center.click", 1),
    ("Urgent: Your WhatsApp account will be deleted in 12 hours due to inactivity. Verify your phone number at http://whatsapp-verify-account.site", 1),
    ("SBI Alert: Dear customer, your debit card reward point of Rs 4,500 is credited. Download SBI reward app to claim: http://bit.ly/sbi-app-apk", 1),
    ("Attention! Your PAN card has been rejected by income tax portal. Link PAN with bank account today to avoid ₹10,000 fine: http://linkpan-bank.online", 1),
    ("Your electricity bill payment of Rs 3,420 failed. Power disconnection team scheduled for 8 PM. Clear bill now: http://tinyurl.com/power-bill-clear", 1),
    ("Package waiting: We have an undelivered parcel for you. Confirm your delivery address now: http://track-shipment-now.buzz", 1),
    ("Urgent notification: Your Google Pay account has received cashback of ₹1,999. Click to transfer to bank: http://gpay-cashback-claim.xyz", 1),
    ("Bank Alert: Your KYC documents have expired. Submit Aadhaar and PAN immediately at http://secure-banking-portal.top or account will be blocked.", 1),

    # --- SAFE: Genuine Bank OTPs (0) ---
    ("482910 is your secret OTP for HDFC Bank NetBanking transaction of INR 1,450.00 at Amazon India. Do NOT share this OTP with anyone, including bank staff.", 0),
    ("Dear SBI Customer, OTP for login to Internet Banking is 839201. Valid for 3 mins. Never share OTP with anyone. State Bank never asks for OTP.", 0),
    ("739104 is the OTP for payment of Rs. 499.00 on Swiggy using ICICI Bank Credit Card ending in 8021. Bank staff will NEVER call asking for OTP.", 0),
    ("Your Axis Bank OTP is 201948 for fund transfer of INR 5,000 to John Doe. Do not disclose OTP/password to anyone.", 0),
    ("Kotak Bank: 618302 is your One Time Password (OTP) for UPI registration. Do not share OTP with anyone. Kotak never calls to ask for OTP.", 0),
    ("902184 is your authentication code for Bank of Baroda mobile banking login. Valid for 5 minutes. Do not share.", 0),
    ("PNB One: Your OTP for password reset is 348219. Please do not share this confidential OTP with anyone.", 0),
    ("IndusInd Bank: OTP for transaction of Rs. 2,100 at Flipkart is 551029. Beware of fraudsters, never disclose OTP.", 0),
    ("Your IDFC FIRST Bank OTP is 918234 for online purchase of Rs 750.00. Valid for 10 minutes. Keep your OTP secret.", 0),
    ("Canara Bank: OTP for generating Green PIN is 102948. Do not share with anyone.", 0),

    # --- SAFE: Genuine Transaction Alerts & Balance Updates (0) ---
    ("Dear Customer, your Account XX9481 has been debited with INR 250.00 on 18-Nov-2026 via UPI reference 6918201948. Avail Bal: INR 14,290.50 - SBI", 0),
    ("HDFC Bank: Alert! Rs 85,000.00 credited to account ending 4920 on 01-Nov-2026 by NEFT-SALARY. Available Balance is INR 92,410.00.", 0),
    ("ICICI Bank: You have spent INR 1,200.00 on your Credit Card XX4019 at Starbucks Mumbai on 15-Nov-2026. Available credit limit: INR 1,48,800.00.", 0),
    ("Axis Bank: ATM cash withdrawal of Rs 4,000.00 on 12-Nov-2026 from ATM ID AXIS0912. Bal: Rs 18,340.00. If not done by you, SMS BLOCK to 5676782.", 0),
    ("Dear Customer, repayment of INR 12,450 towards your Home Loan account HL89201 has been received with thanks. - Kotak Mahindra Bank", 0),
    ("Your bill for ICICI Bank Credit Card ending 7012 is generated. Total Due: Rs 8,420.00. Minimum Due: Rs 500.00. Payment Due Date: 28-Nov-2026.", 0),
    ("Salary of Rs. 65,000.00 has been credited to your SBI account ending in 3312 on 31-Oct-2026. Total Available Balance: Rs. 71,200.00.", 0),
    ("UPI-AutoPay: Recurring payment of Rs 179 for Spotify India has been debited from your HDFC Bank account.", 0),
    ("Dear Customer, Rs 50.00 cashback credited to your PayU wallet for transaction #89120. - PayU", 0),
    ("Account XX1092 debited for Rs 890.00 towards Electricity Bill payment to BESCOM via BBPS. Ref No: 481920194.", 0),

    # --- SAFE: Genuine Courier & Delivery Updates (0) ---
    ("Your Amazon order #402-9182019-1029182 has been shipped via ATS. Expected delivery by tomorrow, 8 PM. Track your package in the official Amazon app.", 0),
    ("Flipkart: Your package containing 'Wireless Bluetooth Headphones' is out for delivery today with wishmaster Rahul. Share OTP 4819 upon receiving.", 0),
    ("Blue Dart: Shipment AWB #8910281920 is delivered to security/reception. Thank you for choosing Blue Dart Express.", 0),
    ("Swiggy: Delivery partner Ramesh is arriving at your location in 5 minutes with your food order. Enjoy your meal!", 0),
    ("Zomato: Order #9182 from Pizza Hut is confirmed and restaurant is preparing your food. Live track on Zomato app.", 0),
    ("Blinkit: Order delivered in 9 minutes! Thank you for shopping with Blinkit. Invoice sent to your registered email.", 0),
    ("Delhivery: Your shipment with tracking number DL91820192 is currently in transit and scheduled for delivery on Monday.", 0),
    ("India Post: Speed Post article EK981204910IN booked on 14/11/2026 at New Delhi GPO has reached destination sorting hub.", 0),

    # --- SAFE: Genuine Service Verification, 2FA & Tickets (0) ---
    ("Google: 849201 is your Google verification code. Never share this code with anyone.", 0),
    ("WhatsApp code: 491-029. You can also tap on the link to verify your phone: v.whatsapp.com/491029. Do not share this code.", 0),
    ("Your Uber verification code is 6109. Never share this code with anyone.", 0),
    ("IRCTC: PNR 2819028190 Train 12952 NDLS-BCT Rajdhani Exp, 22-Nov-2026, 3A, Coach B4, Berth 32 (MB). Confirmed. Have a safe journey.", 0),
    ("IndiGo flight 6E-204 from Delhi to Bengaluru is on schedule. Web check-in is now open. Download boarding pass on goindigo.in.", 0),
    ("Airtel recharge successful! Unlimited 5G + 2GB/day for 84 days activated for number 9876543210. Transaction ID: AT89102910.", 0),
    ("Jio: Your daily data quota of 1.5 GB is 50% consumed on 19-Nov-2026. To get data add-on, open MyJio app.", 0),
    ("Microsoft account security: Use code 719284 to verify your login attempt for user@outlook.com.", 0),
    ("GitHub: Your two-factor authentication code is 492019. It will expire in 10 minutes.", 0),
    ("Apple: Your Apple ID verification code is 801924. Don't share it with anyone.", 0),

    # --- SAFE: Everyday Personal & Work Messages (0) ---
    ("Hey! Are we still meeting for lunch at 1 PM today near Connaught Place?", 0),
    ("Can you please send me the final presentation slides before the 4 PM team sync?", 0),
    ("Mom said dinner will be ready by 8:30 PM. Please bring some bread on your way home.", 0),
    ("Thanks for the birthday wishes! Hope everything is going well with your new job.", 0),
    ("Hey Rahul, I left my laptop charger in conference room B. Could you please check if it is still there?", 0),
    ("The doctor confirmed our appointment for tomorrow at 11:30 AM at Fortis Hospital.", 0),
    ("Happy Diwali to you and your family! Wishing you joy, health, and prosperity.", 0),
    ("Could you review the pull request on GitHub when you have a moment? I added the requested unit tests.", 0),
    ("Hey buddy, let's catch the football match this weekend at my place.", 0),
    ("Good morning sir, submitting the monthly expense report for your approval. Please find the sheet attached.", 0)
]

def train_and_save_model(model_path="scam_model.joblib"):
    print(f"Starting ScamShield ML training with {len(DATASET)} balanced samples...")

    texts = [item[0] for item in DATASET]
    labels = [item[1] for item in DATASET]

    scam_count = sum(labels)
    safe_count = len(labels) - scam_count
    print(f"Dataset breakdown: {scam_count} Scams, {safe_count} Safe messages.")

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    # Pipeline: TF-IDF with unigrams & bigrams + Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            lowercase=True,
            sublinear_tf=True,
            stop_words='english'
        )),
        ('clf', LogisticRegression(
            C=1.5,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Evaluation on Test Split ({len(y_test)} samples):")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Safe", "Scam"]))

    # Save artifact
    output_file = Path(__file__).parent / model_path if "__file__" in globals() else Path(model_path)
    joblib.dump(pipeline, output_file)
    print(f"Successfully trained and saved model artifact to: {output_file.resolve()}\n")
    return pipeline

if __name__ == "__main__":
    train_and_save_model()
