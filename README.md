# SEC IAPD Phase 1 Extractor

Compliance-focused data extractor for public SEC IAPD records, with optional public website email lookup.

## What it does

- Uses the SEC IAPD public compilation feed (`IA_FIRM_SEC_Feed_*.xml.gz`)
- Filters to:
  - `FirmType=Registered` (Investment Adviser)
  - `Status=APPROVED*`
  - U.S.-based main office
  - AUM (`Q5F2C`) between configured bounds
  - Total employees (`TtlEmp`) between configured bounds
- Builds CSV rows for two contact roles per firm:
  - `Chief Compliance Officer`
  - `Executive Officer`
- If email is not present in SEC feed context, crawls only public firm website pages and extracts visible emails

## Output schema

`firm_name,sec_number,aum,employees,city,state,website,contact_name,contact_email,phone,source`

Missing values are written as `N/A`.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python sec_iapd_scraper.py --out advisors.csv
```

Quick verification run:

```bash
python sec_iapd_scraper.py --max-firms 5 --crawl-max-pages 6 --crawl-delay 0.3 --out advisors_sample.csv
```

## Notes

- Website crawling is public-page only; no login or bypass behavior.
- For contact names, if no reliable role-matched name is found, `contact_name` defaults to the role label.
