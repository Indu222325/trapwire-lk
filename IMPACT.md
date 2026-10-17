# TrapWire LK: Impact Statement

## Who is protected

| Group | Size in Sri Lanka | Main scam they receive |
|---|---|---|
| Elderly people and pensioners | 18% of the population is 60+ (Census 2024), about **3.9 million** people | Fake pension suspensions, bank "account blocked" and OTP requests |
| Aswesuma welfare families | About **1.66 million** families paid in January 2026 | Fake "payment pending" links and "registration fee" requests |
| First-time mobile-banking users | Growing quickly as digital payments expand | OTP and PIN theft, fake bank links |

The Central Bank's "Be Scam Proof" campaign (June 2026) names senior citizens, rural communities and low-income households as priority audiences, and asks people to *question, verify and act*. TrapWire is the "verify" step, available in the language people actually receive scams in: Sinhala, Tamil, English, Singlish and Tanglish.

## Harm we prevent

- **Money lost** to fake fees, prizes and account "unblocking".
- **Account takeover** from shared OTPs and PINs.
- **Lost welfare and pension income** when beneficiaries hand over bank details.
- **Silent failure:** most detectors are never checked after launch. TrapWire raises an alert when scammers switch tactics (for example to the upcoming SL-UDI digital ID) and routes uncertain cases to people instead of giving confident wrong answers.

## How many people we could reach

| Stage | Channel | Realistic reach |
|---|---|---|
| Pilot (3 months) | One bank's fraud desk or call centre + Aswesuma help desks at 5 Divisional Secretariats | 5,000 to 20,000 checked messages |
| Year 1 | Bank partners and a WhatsApp "forward to check" number | 100,000+ users |
| Scale | Telco SMS forward short code, integration with the CBSL / Sri Lanka CERT reporting channel | Millions of the 5.5M+ people in the groups above |

## How we will measure it

1. **Scams caught:** share of reported scams TrapWire flags (target: at least 90%; current recall on our synthetic held-out test is 100%).
2. **False alarms:** legitimate messages wrongly flagged (current: 0 on the held-out test set).
3. **Time to adapt:** hours from a new scam wave appearing to an alert and a reviewed, retrained model (demo: one review cycle).
4. **Money protected:** reported losses avoided, estimated with the pilot partner's fraud-case data.

## Path to adoption

1. **Now:** working open-source service (MIT licence), Docker image, monitoring and human-review workflow.
2. **Next 3 months:** pilot with a bank fraud team or welfare help desk; replace synthetic data with consented, anonymised scam reports; staff act as the reviewers.
3. **Then:** WhatsApp and SMS-forwarding front end for the public; share drift alerts with CBSL and Sri Lanka CERT so awareness campaigns follow new scam tactics within days, not months.

**Cost:** runs on a single small CPU server (no GPU, no paid LLM calls), so a public tool is affordable for a bank, telco or government partner.
