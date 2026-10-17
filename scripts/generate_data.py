"""
Generate the synthetic TrapWire LK dataset.

All messages are synthetic: bank names are fictional, links use the reserved
".example" domain, and phone numbers are masked. Nothing here is real personal
data, and none of these messages may be sent to real people (Rule 2.6).

Outputs (CSV, columns: text, label, scam_type, lang, template_id):
  data/train.csv    training set
  data/test.csv     held-out test set (includes templates never seen in training)
  data/shifted.csv  drift set: a NEW scam target (SL-UDI digital ID), heavier
                    Tanglish/Singlish mixing and obfuscated spellings
"""
import random
import re
from pathlib import Path

import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

BANKS = ["Lanka Trust Bank", "Ceylon Union Bank", "Island Savings Bank", "LTB", "CUB", "ISB"]
LINK_WORDS = ["secure", "verify", "update", "lk-help", "account", "refund", "claim", "gov-lk", "pay"]


def link(rng):
    return f"http://{rng.choice(LINK_WORDS)}-{rng.choice(LINK_WORDS)}.example/{rng.randint(100, 999)}"


def phone(rng):
    return f"07{rng.randint(0, 8)}-xxx-{rng.randint(1000, 9999)}"


def fill(t, rng):
    return t.format(
        bank=rng.choice(BANKS),
        link=link(rng),
        phone=phone(rng),
        amount=f"{rng.choice([2500, 5000, 7500, 10000, 15000, 25000, 48000]):,}",
        fee=f"{rng.choice([250, 500, 750, 1000, 1500, 2000]):,}",
        prize=f"{rng.choice([500000, 1000000, 2500000, 5000000]):,}",
        otp=rng.randint(100000, 999999),
        time=rng.choice(["9.00", "10.30", "2.00", "4.15"]),
    )


# ---------------------------------------------------------------------------
# Training-distribution templates: class -> lang -> list of templates
# ---------------------------------------------------------------------------
T = {
    "bank_impersonation": {
        "en": [
            "Dear customer, your {bank} account has been temporarily suspended. Verify immediately at {link}",
            "{bank}: Your debit card is blocked due to suspicious activity. Reply with the OTP sent to you to unblock.",
            "URGENT: A transaction of Rs. {amount} was requested from your account. If this was not you, cancel at {link}",
            "Your online banking access expires today. Update your details now at {link} to avoid closure.",
            "{bank} security team: we detected a login from a new device. Call {phone} and share your OTP to secure your account.",
        ],
        "si": [
            "ඔබගේ {bank} ගිණුම තාවකාලිකව අත්හිටුවා ඇත. වහාම තහවුරු කිරීමට {link} වෙත පිවිසෙන්න.",
            "ඔබගේ කාඩ්පත අවහිර වී ඇත. OTP අංකය මෙම අංකයට එවන්න {phone}.",
            "{bank}: ඔබගේ ගිණුමෙන් රු. {amount} ක් අඩු කිරීමට ඉල්ලීමක් ලැබී ඇත. ඔබ නොවේ නම් {link} ක්ලික් කරන්න.",
            "ඔබගේ ඔන්ලයින් බැංකු සේවාව අද දින අවසන් වේ. යාවත්කාලීන කිරීමට {link}",
        ],
        "ta": [
            "உங்கள் {bank} கணக்கு தற்காலிகமாக முடக்கப்பட்டுள்ளது. உடனே சரிபார்க்க {link} ஐ அழுத்தவும்.",
            "உங்கள் அட்டை தடுக்கப்பட்டுள்ளது. OTP எண்ணை {phone} க்கு அனுப்பவும்.",
            "உங்கள் கணக்கிலிருந்து ரூ. {amount} எடுக்க கோரிக்கை வந்துள்ளது. நீங்கள் இல்லையென்றால் {link} கிளிக் செய்யவும்.",
        ],
        "singlish": [
            "Oyage {bank} account eka block karala thiyenne. Ikmanata verify karanna {link}",
            "Card eka block wela. Unblock karanna oyata awa OTP eka me number ekata ewanna {phone}",
            "Oyage account eken Rs. {amount} ganna request ekak awa. Oya nemei nam {link} click karanna",
            "Online banking eka ada iwara wenawa. Details update karanna {link} nathnam account eka close wenawa",
        ],
        "tanglish": [
            "Unga {bank} account block aagiduchu. Udane verify panna {link}",
            "Card block aagiduchu. OTP ah intha number ku anupunga {phone}",
        ],
    },
    "welfare_scam": {
        "en": [
            "Aswesuma payment of Rs. {amount} is pending for you. Enter your details at {link} to receive it today.",
            "You have been selected as an Aswesuma beneficiary. Send the registration fee of Rs. {fee} to {phone}.",
            "Your Aswesuma benefit has been stopped. Reply with your bank PIN to restart payments.",
            "Government relief grant of Rs. {amount} approved for your family. Claim before midnight at {link}",
        ],
        "si": [
            "අස්වැසුම දීමනාව රු. {amount} ලබා ගැනීමට ඔබගේ විස්තර {link} හි ඇතුලත් කරන්න.",
            "ඔබ අස්වැසුම ප්‍රතිලාභියෙක් ලෙස තෝරාගෙන ඇත. ලියාපදිංචි ගාස්තුව රු. {fee} {phone} අංකයට එවන්න.",
            "අස්වැසුම ගෙවීම අත්හිටුවා ඇත. නැවත ආරම්භ කිරීමට ඔබගේ බැංකු PIN අංකය සමඟ පිළිතුරු දෙන්න.",
        ],
        "ta": [
            "அஸ்வெசும கொடுப்பனவு ரூ. {amount} பெற உங்கள் விவரங்களை {link} இல் பதிவு செய்யவும்.",
            "நீங்கள் அஸ்வெசும பயனாளியாக தேர்வு செய்யப்பட்டுள்ளீர்கள். பதிவு கட்டணம் ரூ. {fee} ஐ {phone} க்கு அனுப்பவும்.",
            "அஸ்வெசும கொடுப்பனவு நிறுத்தப்பட்டுள்ளது. மீண்டும் தொடங்க உங்கள் வங்கி PIN எண்ணுடன் பதில் அனுப்பவும்.",
        ],
        "singlish": [
            "Aswesuma salli Rs. {amount} oyata pending. Ada ma ganna details {link} eke danna",
            "Oyawa Aswesuma ekata select karala. Registration fee Rs. {fee} {phone} ekata ewanna",
            "Aswesuma payment eka nawaththala. Ayeth patan ganna bank PIN eka reply karanna",
        ],
        "tanglish": [
            "Aswesuma panam Rs. {amount} ungalukku pending. Details {link} la podunga",
        ],
    },
    "pension_scam": {
        "en": [
            "Pension increase of Rs. {amount} approved. Enter your NIC number at {link} to activate it.",
            "Your pension has been suspended. Call {phone} immediately to avoid cancellation.",
            "Pensioners must update their pension account before month end or payments will stop. Update at {link}",
            "Department notice: arrears of Rs. {amount} due to you. Pay a processing fee of Rs. {fee} to release.",
        ],
        "si": [
            "විශ්‍රාම වැටුප් වැඩිවීම රු. {amount} ලබා ගැනීමට ඔබගේ ජාතික හැඳුනුම්පත් අංකය {link} හි ඇතුලත් කරන්න.",
            "ඔබගේ විශ්‍රාම වැටුප නවත්වා ඇත. වහාම {phone} අමතන්න.",
            "විශ්‍රාම වැටුප් ගිණුම යාවත්කාලීන නොකළහොත් ලබන මාසයේ ගෙවීම නතර වේ. {link}",
        ],
        "ta": [
            "ஓய்வூதிய உயர்வு ரூ. {amount} பெற உங்கள் தேசிய அடையாள அட்டை எண்ணை {link} இல் உள்ளிடவும்.",
            "உங்கள் ஓய்வூதியம் நிறுத்தப்பட்டுள்ளது. உடனே {phone} ஐ அழைக்கவும்.",
            "ஓய்வூதிய கணக்கை புதுப்பிக்காவிட்டால் அடுத்த மாத கொடுப்பனவு நிறுத்தப்படும். {link}",
        ],
        "singlish": [
            "Pension eka Rs. {amount} kin wadi karala. Activate karanna NIC number eka {link} eke danna",
            "Oyage pension eka nawaththala. Ikmanata {phone} ekata call karanna",
            "Pension account eka update kale nathnam ilanga masse salli nawathinawa. {link}",
        ],
        "tanglish": [
            "Unga pension stop aagiduchu. Udane {phone} ku call pannunga",
        ],
    },
    "parcel_scam": {
        "en": [
            "Your parcel is held at customs. Pay Rs. {fee} to release it at {link}",
            "Delivery failed due to an incorrect address. Update your address at {link} within 24 hours.",
            "You have received a gift from abroad. Pay the tax of Rs. {fee} to collect it.",
            "Courier notice: your package will be returned today. Confirm and pay redelivery fee at {link}",
        ],
        "si": [
            "ඔබගේ පාර්සලය රේගුවේ රඳවා ඇත. රු. {fee} ගෙවා නිදහස් කරගන්න {link}",
            "ලිපිනය වැරදි නිසා පාර්සලය බෙදා හැරීමට නොහැක. යාවත්කාලීන කරන්න {link}",
            "ඔබට විදේශයෙන් තෑග්ගක් ලැබී ඇත. බදු ගාස්තුව රු. {fee} ගෙවන්න.",
        ],
        "ta": [
            "உங்கள் பார்சல் சுங்கத்தில் நிறுத்தப்பட்டுள்ளது. ரூ. {fee} செலுத்தி விடுவிக்கவும் {link}",
            "தவறான முகவரி காரணமாக பார்சலை வழங்க முடியவில்லை. புதுப்பிக்கவும் {link}",
            "வெளிநாட்டிலிருந்து உங்களுக்கு பரிசு வந்துள்ளது. வரி கட்டணம் ரூ. {fee} செலுத்தவும்.",
        ],
        "singlish": [
            "Oyage parcel eka customs eke hira wela. Rs. {fee} gewala release karaganna {link}",
            "Address eka waradi nisa parcel eka deliver karanna ba. Update karanna {link}",
            "Pitarata idan oyata gift ekak awa. Tax eka Rs. {fee} gewala ganna",
        ],
        "tanglish": [
            "Unga parcel customs la irukku. Rs. {fee} kattunga {link}",
        ],
    },
    "lottery_scam": {
        "en": [
            "Congratulations! You have won Rs. {prize} in the mobile lucky draw. Claim at {link}",
            "Your number won the national SMS lottery. Send the processing fee of Rs. {fee} to receive your prize.",
            "You are today's lucky winner. Call {phone} to collect your cash prize.",
            "WINNER ALERT: Rs. {prize} reward is waiting. Share your bank details to transfer the money.",
        ],
        "si": [
            "සුබ පැතුම්! ඔබ රු. {prize} ක ත්‍යාගයක් දිනා ඇත. ලබා ගැනීමට {link}",
            "ඔබගේ දුරකථන අංකය ලොතරැයියෙන් ජය ගෙන ඇත. සැකසුම් ගාස්තුව රු. {fee} එවන්න.",
            "ඔබ අද දිනයේ වාසනාවන්ත ජයග්‍රාහකයා. ත්‍යාගය ලබා ගැනීමට {phone} අමතන්න.",
        ],
        "ta": [
            "வாழ்த்துக்கள்! நீங்கள் ரூ. {prize} பரிசு வென்றுள்ளீர்கள். பெற {link}",
            "உங்கள் தொலைபேசி எண் லாட்டரியில் வென்றுள்ளது. செயலாக்க கட்டணம் ரூ. {fee} அனுப்பவும்.",
            "இன்றைய அதிர்ஷ்ட வெற்றியாளர் நீங்கள். பரிசைப் பெற {phone} ஐ அழைக்கவும்.",
        ],
        "singlish": [
            "Subapathum! Oya Rs. {prize} dinala. Ganna {link}",
            "Oyage number eka lottery eken dinala. Processing fee Rs. {fee} ewanna",
            "Ada lucky winner oya. Prize eka ganna {phone} ekata call karanna",
        ],
        "tanglish": [
            "Vaazhthukkal! Neenga Rs. {prize} jeyichirukeenga. Claim panna {link}",
        ],
    },
    "legit": {
        "en": [
            "Your OTP is {otp}. Do not share it with anyone. The bank will never ask for your OTP.",
            "Rs. {amount} has been credited to your account. Check your balance in your bank app.",
            "Reminder: your doctor's appointment is tomorrow at {time}. Please arrive on time.",
            "Hi, I will be home late today. Please keep dinner for me.",
            "Your electricity bill of Rs. {amount} is due on the 25th. Pay at any authorised outlet.",
            "Aswesuma payments for this month will be credited to beneficiary accounts this week. No officer will ask for your PIN.",
            "Your parcel has been delivered. Thank you for shopping with us.",
            "Meeting moved to {time} tomorrow. See you in the office.",
        ],
        "si": [
            "ඔබගේ OTP අංකය {otp}. මෙය කිසිවෙකු සමඟ බෙදා නොගන්න. බැංකුව කිසිවිටෙක OTP අසන්නේ නැත.",
            "ඔබගේ ගිණුමට රු. {amount} ක් බැර කරන ලදී. ශේෂය පරීක්ෂා කිරීමට ඔබගේ බැංකු ඇප් එක භාවිතා කරන්න.",
            "අම්මා, මම අද හවස ගෙදර එනවා. කෑම ලෑස්ති කරන්න.",
            "ඔබගේ වෛද්‍ය හමුව හෙට {time}ට. කරුණාකර වේලාවට පැමිණෙන්න.",
            "අස්වැසුම ගෙවීම් මෙම සතියේ ඔබගේ ගිණුමට බැර වේ. කිසිවෙකු ඔබගෙන් PIN අංකය අසන්නේ නැත.",
        ],
        "ta": [
            "உங்கள் OTP {otp}. இதை யாருடனும் பகிர வேண்டாம். வங்கி ஒருபோதும் OTP கேட்காது.",
            "உங்கள் கணக்கில் ரூ. {amount} வரவு வைக்கப்பட்டது. இருப்பை வங்கி செயலியில் பார்க்கவும்.",
            "அம்மா, நான் இன்று மாலை வீட்டுக்கு வருகிறேன்.",
            "உங்கள் மருத்துவ சந்திப்பு நாளை {time} மணிக்கு.",
        ],
        "singlish": [
            "Machan ada match ekata enawada? {time} ta patan ganna",
            "Amma mama ada rae wela yanawa, kema thiyanna",
            "Oyage OTP eka {otp}. Kawuruwath ekka share karanna epa",
            "Rs. {amount} oyage account ekata credit una. Bank app eken balance eka balanna",
            "Heta class eka {time} ta. Notes tika genna amathaka karanna epa",
        ],
        "tanglish": [
            "Machan naalaikku {time} ku meet pannalaam",
            "Amma naan late ah varuven, saapadu vechirunga",
        ],
    },
}

# ---------------------------------------------------------------------------
# Drift templates: a NEW impersonation target that did not exist in training
# (the planned SL-UDI national digital ID), in heavier code-mixed language.
# ---------------------------------------------------------------------------
DRIFT = {
    "digital_id_scam": {
        "en": [
            "SL-UDI registration closing today. Complete your digital ID biometric update at {link} or lose access to government services.",
            "Your new digital ID has been generated. Scan the QR code and confirm your face scan at {link}",
            "SL-UDI notice: your eID activation failed. Pay Rs. {fee} verification charge to {phone}.",
            "Digital identity wallet locked. Re-link your bank account to your SL-UDI at {link}",
        ],
        "singlish": [
            "SL-UDI digital ID eka register karanne ada antima dawasa. Biometric update eka {link} eke karanna",
            "Oyage aluth digital ID eka ready. QR eka scan karala face scan eka confirm karanna {link}",
            "eID activate wenne na. Verification charge Rs. {fee} {phone} ekata ewanna",
            "Digital ID wallet eka lock wela. Bank account eka SL-UDI ekata link karanna {link}",
        ],
        "tanglish": [
            "SL-UDI digital ID register panna innaiku last day. Biometric update {link} la pannunga",
            "Unga pudhu digital ID ready. QR scan panni face confirm pannunga {link}",
            "eID activate aagala. Verification charge Rs. {fee} {phone} ku anupunga",
        ],
        "si": [
            "SL-UDI ඩිජිටල් හැඳුනුම්පත ලියාපදිංචිය අද අවසන්. ජෛවමිතික යාවත්කාලීන කිරීම {link} හි කරන්න.",
            "ඔබගේ නව ඩිජිටල් හැඳුනුම්පත සූදානම්. QR කේතය ස්කෑන් කර මුහුණ තහවුරු කරන්න {link}",
        ],
        "ta": [
            "SL-UDI டிஜிட்டல் அடையாள பதிவு இன்று முடிகிறது. பயோமெட்ரிக் புதுப்பிப்பை {link} இல் செய்யவும்.",
            "உங்கள் புதிய டிஜிட்டல் ஐடி தயார். QR குறியீட்டை ஸ்கேன் செய்து முகத்தை உறுதிப்படுத்தவும் {link}",
        ],
    },
    "legit": {
        "en": [
            "SL-UDI registration is free and done only at your Divisional Secretariat. No fee is collected by SMS.",
            "Your digital ID appointment is confirmed for {time} at the Divisional Secretariat. Bring your old NIC.",
        ],
        "singlish": [
            "SL-UDI register karanna salli ona na, Divisional Secretariat ekata yanna witharai",
            "Machan digital ID ekata appointment eka heta {time} ta, ekata yamuda",
        ],
        "tanglish": [
            "SL-UDI registration free, Divisional Secretariat la mattum pannalaam",
            "Machan digital ID appointment naalaikku {time} ku",
        ],
    },
}

LEET = {"a": "@", "o": "0", "i": "1", "e": "3", "s": "$"}
EMOJI = ["🚨", "⚠️", "✅", "🎁", "💰", "👉", "🔒"]


def obfuscate(text, rng):
    """Scammer evasion tricks: leetspeak, emoji, spaced keywords."""
    out = []
    for w in text.split(" "):
        if w.startswith("http") or not w.isascii():
            out.append(w)
            continue
        r = rng.random()
        if r < 0.18:
            w = "".join(LEET.get(c, c) if rng.random() < 0.6 else c for c in w)
        elif r < 0.24 and len(w) > 3:
            w = " ".join(w)
        out.append(w)
    t = " ".join(out)
    if rng.random() < 0.7:
        t = f"{rng.choice(EMOJI)} {t} {rng.choice(EMOJI)}"
    return t


def is_scam(cls):
    return 0 if cls == "legit" else 1


def build_rows(spec, n_per_template, rng, template_filter=None, prefix=""):
    rows = []
    for cls, langs in spec.items():
        for lang, temps in langs.items():
            for i, t in enumerate(temps):
                tid = f"{prefix}{cls}:{lang}:{i}"
                if template_filter and not template_filter(cls, lang, i, len(temps)):
                    continue
                for _ in range(n_per_template):
                    rows.append({"text": fill(t, rng), "label": is_scam(cls), "scam_type": cls,
                                 "lang": lang, "template_id": tid})
    return rows


def main():
    rng = random.Random(SEED)
    DATA.mkdir(exist_ok=True)

    # Hold out the LAST template of every (class, language) group that has 3+
    # templates. Those templates appear ONLY in the test set, so the test score
    # measures generalisation to unseen wording, not memorised templates.
    def seen(cls, lang, i, n):
        return not (n >= 3 and i == n - 1)

    def unseen(cls, lang, i, n):
        return n >= 3 and i == n - 1

    seen_rows = build_rows(T, 40, rng, seen)
    rng.shuffle(seen_rows)
    df_seen = pd.DataFrame(seen_rows)
    # 80/20 split of seen-template messages, plus all unseen-template messages in test
    test_seen = df_seen.sample(frac=0.2, random_state=SEED)
    train = df_seen.drop(test_seen.index)
    test_unseen = pd.DataFrame(build_rows(T, 25, rng, unseen))
    test = pd.concat([test_seen, test_unseen]).sample(frac=1, random_state=SEED)

    # Shifted set: new target + obfuscated old scams + some ordinary traffic
    drift_rows = build_rows(DRIFT, 30, rng, prefix="drift:")
    old = build_rows(T, 4, rng, prefix="obf:")
    for r in old:
        if r["label"] == 1:
            r["text"] = obfuscate(r["text"], rng)
    shifted = pd.DataFrame(drift_rows + old).sample(frac=1, random_state=SEED)

    cols = ["text", "label", "scam_type", "lang", "template_id"]
    train[cols].to_csv(DATA / "train.csv", index=False)
    test[cols].to_csv(DATA / "test.csv", index=False)
    shifted[cols].to_csv(DATA / "shifted.csv", index=False)

    for name, d in [("train", train), ("test", test), ("shifted", shifted)]:
        print(f"{name:8s} {len(d):5d} rows | scam share {d.label.mean():.2f} | "
              f"langs { {k: int(v) for k, v in d.lang.value_counts().items()} }")


if __name__ == "__main__":
    main()
