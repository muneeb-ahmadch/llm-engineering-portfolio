# Ground truth: what the test PDFs actually say

This is the **verbatim source text** of the three PDFs in [`test-corpus/`](test-corpus/).
Keep it out of the ingest path (the loader should target `test-corpus/*.pdf`, not this
file). It exists so I can (a) regenerate the PDFs and (b) **check the faithfulness judge's
verdicts against what the corpus really contains.** You cannot grade faithfulness without a
ground truth; this is it.

Regenerate any PDF with macOS's built-in filter (no install):

```bash
/usr/sbin/cupsfilter source.txt > test-corpus/name.pdf
```

---

## insureelm-claims-policy.pdf  (CLM-2026)

- Cooling-off period: **14 calendar days** from issue date, full refund if no claim made.
- Claim reference format: **CLM-YYYY-NNNNNN**.
- First acknowledgement SLA: **2 business days**; standard resolution: **10 business days**;
  complex resolution: **25 business days**.
- Standard health-claim excess: **PKR 5,000**. Travel claims: **no excess**.
- Emergency claims line hours: **06:00–22:00 PKT**, 7 days a week.
- "Complex" = hospitalisation over PKR 200,000, third-party liability, or suspected
  non-disclosure → routed to the Claims Review Committee.
- Escalation contact: **Priya Nandakumar, Claims Escalation Lead** (responds within 5
  business days). Further appeal → Insurance Ombudsman (out of scope of the doc).

## insureelm-coverage-tiers.pdf  (PRD-HLT-2026)

- Tiers: **Bronze / Silver / Gold**.
- Monthly premium: **1,200 / 2,800 / 4,500 PKR**.
- Annual coverage limit: **250,000 / 750,000 / 2,000,000 PKR**.
- Initial waiting period: **90 / 45 / 15 days**.
- Room entitlement: general ward / semi-private / private.
- International cover ("GlobalAssist"): **Gold only**, up to **PKR 5,000,000** per event,
  trips ≤ 30 consecutive days outside Pakistan.
- **Excluded from every tier:** dental, cosmetic, declared pre-existing conditions. **No
  dental add-on exists.**
- Health tiers only. Motor, life and property are documented elsewhere, not in this sheet.

## insureelm-employee-handbook.pdf  (HR-04 excerpt)

- Remote work: up to **3 days/week** with manager approval; **core hours 11:00–16:00 PKT**;
  fully-remote needs Director sign-off, reviewed every 6 months.
- VPN: **ElmGate** (split tunnelling disabled). Ticketing: **Rootline** (email to IT is
  untracked).
- Phishing reported within **1 hour** via the "Report Phish" button in Rootline.
- Passwords rotate every **90 days**; last 5 blocked from reuse.
- On exit: hardware returned within **5 business days**; ElmGate/Rootline access revoked
  **18:00 PKT** on the final day.
- This is an *excerpt*. Leave, payroll and grievance are HR-01…HR-03, **not included here**.
