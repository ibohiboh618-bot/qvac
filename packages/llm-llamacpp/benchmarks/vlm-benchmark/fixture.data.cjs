/* eslint-disable */
'use strict'
// VLM benchmark fixture. Bootstrapped by build-fixture.cjs, then
// HAND-CURATED: VQA golds reduced to distinct correct answers + synonyms;
// OCR tasks added — ocr-small (curated proverb phrase images) + ocr-page (getomni-ai/
// ocr-benchmark, MIT, full-document markdown transcription), scored by CER/WER/BLEU.
// Images are S3-only (see fixture/README.md). A build-fixture regen overwrites this file.
module.exports = {
  "tasks": [
    "textvqa",
    "vizwiz",
    "gqa",
    "docvqa",
    "ai2d",
    "ocr-small",
    "ocr-page",
    "tiling-perf",
    "disc-ocr"
  ],
  "samplesPerTask": 5,
  "items": [
    {
      "id": "textvqa_0",
      "task": "textvqa",
      "metric": "vqa",
      "prompt": "who was the photographer?\nAnswer the question using a single word or phrase.",
      "gold": [
        "philippe molitor"
      ],
      "image": "vlmx-textvqa_0.jpg",
      "width": 1024,
      "height": 681
    },
    {
      "id": "textvqa_1",
      "task": "textvqa",
      "metric": "vqa",
      "prompt": "what is the year on the calender?\nAnswer the question using a single word or phrase.",
      "gold": [
        "2010"
      ],
      "image": "vlmx-textvqa_1.jpg",
      "width": 1024,
      "height": 768
    },
    {
      "id": "textvqa_2",
      "task": "textvqa",
      "metric": "vqa",
      "prompt": "what is the largest measurement we can see on this ruler?\nAnswer the question using a single word or phrase.",
      "gold": [
        "50"
      ],
      "image": "vlmx-textvqa_2.jpg",
      "width": 1024,
      "height": 768
    },
    {
      "id": "textvqa_3",
      "task": "textvqa",
      "metric": "vqa",
      "prompt": "how much is the coin worth?\nAnswer the question using a single word or phrase.",
      "gold": [
        "25 paise",
        "25"
      ],
      "image": "vlmx-textvqa_3.jpg",
      "width": 1024,
      "height": 768
    },
    {
      "id": "textvqa_4",
      "task": "textvqa",
      "metric": "vqa",
      "prompt": "what is the 3 letter word to the left of casa in the text?\nAnswer the question using a single word or phrase.",
      "gold": [
        "tua"
      ],
      "image": "vlmx-textvqa_4.jpg",
      "width": 1024,
      "height": 765
    },
    {
      "id": "vizwiz_0",
      "task": "vizwiz",
      "metric": "vqa",
      "prompt": "Is there text or any identifying identification on this side of the card?\nAnswer the question using a single word or phrase.",
      "gold": [
        "no",
        "none"
      ],
      "image": "vlmx-vizwiz_0.jpg",
      "width": 360,
      "height": 480
    },
    {
      "id": "vizwiz_1",
      "task": "vizwiz",
      "metric": "vqa",
      "prompt": "What title is this?\nAnswer the question using a single word or phrase.",
      "gold": [
        "every now and then"
      ],
      "image": "vlmx-vizwiz_1.jpg",
      "width": 360,
      "height": 480
    },
    {
      "id": "vizwiz_2",
      "task": "vizwiz",
      "metric": "vqa",
      "prompt": "Is this computer actually on a state where it's still sorting itself out? And getting ready to boot? Or, is it already booted? \nAnswer the question using a single word or phrase.",
      "gold": [
        "already booted",
        "booted"
      ],
      "image": "vlmx-vizwiz_2.jpg",
      "width": 484,
      "height": 648
    },
    {
      "id": "vizwiz_3",
      "task": "vizwiz",
      "metric": "vqa",
      "prompt": "What color is this?\nAnswer the question using a single word or phrase.",
      "gold": [
        "purple",
        "magenta",
        "burgundy"
      ],
      "image": "vlmx-vizwiz_3.jpg",
      "width": 768,
      "height": 1024
    },
    {
      "id": "vizwiz_4",
      "task": "vizwiz",
      "metric": "vqa",
      "prompt": "What is the focused button title?\nAnswer the question using a single word",
      "gold": [
        "Restore"
      ],
      "image": "vlmx-vizwiz_4.jpg",
      "width": 768,
      "height": 1024
    },
    {
      "id": "gqa_0",
      "task": "gqa",
      "metric": "vqa",
      "prompt": "Who is wearing the shirt?\nAnswer the question using a single word or phrase.",
      "gold": [
        "girl",
        "woman",
        "lady"
      ],
      "image": "vlmx-gqa_0.jpg",
      "width": 500,
      "height": 331
    },
    {
      "id": "gqa_1",
      "task": "gqa",
      "metric": "vqa",
      "prompt": "What is hanging above the chalkboard?\nAnswer the question using a single word or phrase.",
      "gold": [
        "picture",
        "painting",
        "artwork"
      ],
      "image": "vlmx-gqa_1.jpg",
      "width": 427,
      "height": 640
    },
    {
      "id": "gqa_2",
      "task": "gqa",
      "metric": "vqa",
      "prompt": "Are both the phone and the coffee cup the same color?\nAnswer the question using a single word or phrase.",
      "gold": [
        "yes"
      ],
      "image": "vlmx-gqa_2.jpg",
      "width": 640,
      "height": 425
    },
    {
      "id": "gqa_3",
      "task": "gqa",
      "metric": "vqa",
      "prompt": "Which color is the shirt?\nAnswer the question using a single word or phrase.",
      "gold": [
        "white"
      ],
      "image": "vlmx-gqa_3.jpg",
      "width": 640,
      "height": 428
    },
    {
      "id": "gqa_4",
      "task": "gqa",
      "metric": "vqa",
      "prompt": "Are the cabinets below the stove wooden and open?\nAnswer the question using a single word or phrase.",
      "gold": [
        "no"
      ],
      "image": "vlmx-gqa_4.jpg",
      "width": 640,
      "height": 427
    },
    {
      "id": "docvqa_0",
      "task": "docvqa",
      "metric": "anls",
      "prompt": "In the form, what comes first, 'to' or 'from'?\nAnswer the question using a single word or phrase.",
      "gold": [
        "to"
      ],
      "image": "vlmx-docvqa_0.jpg",
      "width": 646,
      "height": 440
    },
    {
      "id": "docvqa_1",
      "task": "docvqa",
      "metric": "anls",
      "prompt": "Which description comes under the label \"NAME\"?\nAnswer the question using a single word or phrase.",
      "gold": [
        "address"
      ],
      "image": "vlmx-docvqa_1.jpg",
      "width": 904,
      "height": 725
    },
    {
      "id": "docvqa_2",
      "task": "docvqa",
      "metric": "anls",
      "prompt": "What is the NET WT.?\nAnswer the question using a single word or phrase.",
      "gold": [
        "10 pounds",
        "10 lbs"
      ],
      "image": "vlmx-docvqa_2.jpg",
      "width": 957,
      "height": 990
    },
    {
      "id": "docvqa_3",
      "task": "docvqa",
      "metric": "anls",
      "prompt": "To which department the letter should be sent ?\nAnswer the question using a single word or phrase.",
      "gold": [
        "sales department",
        "sales"
      ],
      "image": "vlmx-docvqa_3.jpg",
      "width": 957,
      "height": 990
    },
    {
      "id": "docvqa_4",
      "task": "docvqa",
      "metric": "anls",
      "prompt": "What is the name of the late queen?\nAnswer the question using a single word or phrase.",
      "gold": [
        "queen mary",
        "mary"
      ],
      "image": "vlmx-docvqa_4.jpg",
      "width": 1024,
      "height": 834
    },
    {
      "id": "ai2d_0",
      "task": "ai2d",
      "metric": "mc",
      "prompt": "What stage comes after egg?\nA. beetle\nB. caterpillar\nC. pupa\nD. mealworm\nAnswer with ONLY the letter (for example, A) of the correct option, and nothing else.",
      "gold": [
        "D"
      ],
      "image": "vlmx-ai2d_0.jpg",
      "width": 350,
      "height": 300
    },
    {
      "id": "ai2d_1",
      "task": "ai2d",
      "metric": "mc",
      "prompt": "Which label shows the first stage of the life cycle?\nA. J\nB. C\nC. E\nD. F\nAnswer with ONLY the letter (for example, A) of the correct option, and nothing else.",
      "gold": [
        "A"
      ],
      "image": "vlmx-ai2d_1.jpg",
      "width": 500,
      "height": 361
    },
    {
      "id": "ai2d_2",
      "task": "ai2d",
      "metric": "mc",
      "prompt": "Based on the given diagram answer the following question. What would happen to the elephant seal population if there were no whales?\nA. Increase\nB. None of these\nC. Neither increase nor decrease\nD. Decrease\nAnswer with ONLY the letter (for example, A) of the correct option, and nothing else.",
      "gold": [
        "A"
      ],
      "image": "vlmx-ai2d_2.jpg",
      "width": 576,
      "height": 396
    },
    {
      "id": "ai2d_3",
      "task": "ai2d",
      "metric": "mc",
      "prompt": "Larvae turn into what form?\nA. Ovums\nB. Exoskeletons\nC. Eggs\nD. Adults\nAnswer with ONLY the letter (for example, A) of the correct option, and nothing else.",
      "gold": [
        "D"
      ],
      "image": "vlmx-ai2d_3.jpg",
      "width": 591,
      "height": 688
    },
    {
      "id": "ai2d_4",
      "task": "ai2d",
      "metric": "mc",
      "prompt": "From the above food web diagram, what will cause moose to increase\nA. decrease in evergreen\nB. decrease in bobcar\nC. increase in bobcat\nD. increase in evergreen\nAnswer with ONLY the letter (for example, A) of the correct option, and nothing else.",
      "gold": [
        "D"
      ],
      "image": "vlmx-ai2d_4.jpg",
      "width": 864,
      "height": 592
    },
    {
      "id": "ocr-page_0",
      "task": "ocr-page",
      "metric": "ocr",
      "prompt": "Transcribe all the text in this document as Markdown. Output only the transcription.",
      "gold": [
        "Ezequiel Kilback93293 Cedar RoadJedediahport, Kansas 01204-1201\n\n![Wells Fargo](image)\n\n2025-02-12\n\nPAY TO THE ORDER OF\n\nWilderman and Sons\n\n$18,539.51\n\nEighteen Thousand Five Hundred Thirty Nine and 51/100\n\nDOLLARS\n\nMEMO\n\nEquipment purchase\n\nEzequiel Kilback\n\n⑆604632435⑆  9191333178  ⑈3218⑈\n\n\\------------------------------------------------\n"
      ],
      "image": "vlmx-ocrpage_0.jpg",
      "width": 1872,
      "height": 912
    },
    {
      "id": "ocr-page_1",
      "task": "ocr-page",
      "metric": "ocr",
      "prompt": "Transcribe all the text in this document as Markdown. Output only the transcription.",
      "gold": [
        "# Real Estate Market Dynamics\n\n| City        | Year | Transactions (Total) | Permits Filed | Data Centers | Hospitality |\n| ----------- | ---- | -------------------- | ------------- | ------------ | ----------- |\n| Charlotte   | 2020 | $154M                | 103           | $74.3M       | $79.8M      |\n| New York    | 2020 | $190M                | 53            | $29.2M       | $160.8M     |\n| Seattle     | 2020 | $219M                | 76            | $170.3M      | $48.7M      |\n| Washington  | 2020 | $157M                | 77            | $81.8M       | $75.2M      |\n| San Antonio | 2020 | $44M                 | 99            | $14.3M       | $29.7M      |\n| San Jose    | 2020 | $291M                | 440           | $162.6M      | $128.4M     |\n\n### Data Centers\n\nAvg: $22.0M\n\nMedian: $19.5M\n\n### Hospitality\n\nAvg: $18.3M\n\nMedian: $20.0M\n\n### Transactions by Quarter\n\n!\\[Bar chart showing Transactions by Quarter\\]\n\n| date    | Transaction Amount ($M) | category     |\n| ------- | ----------------------- | ------------ |\n| Q1-2020 | 15                      | Data Centers |\n| Q2-2020 | 35                      | Data Centers |\n| Q3-2020 | 24                      | Data Centers |\n| Q4-2020 | 14                      | Data Centers |\n| Q1-2020 | 18                      | Hospitality  |\n| Q2-2020 | 10                      | Hospitality  |\n| Q3-2020 | 23                      | Hospitality  |\n| Q4-2020 | 22                      | Hospitality  |\n\nNew Location Growth\n\n![Crown Castle Logo](image)\n\nPage 22 / 36\n"
      ],
      "image": "vlmx-ocrpage_1.jpg",
      "width": 1918,
      "height": 1716
    },
    {
      "id": "ocr-page_2",
      "task": "ocr-page",
      "metric": "ocr",
      "prompt": "Transcribe all the text in this document as Markdown. Output only the transcription.",
      "gold": [
        "| **Attributes**     | **P (%)** | **R (%)** | **F1 (%)** |\n| ------------------ | --------- | --------- | ---------- |\n| Frame Color        | 63.16     | 48.00     | 54.55      |\n| Lenses Color       | 64.29     | 40.91     | 50.00      |\n| Shell Material     | 54.05     | 44.44     | 48.78      |\n| Wheel Material     | 70.59     | 37.50     | 48.98      |\n| Product Type       | 64.86     | 43.29     | 51.92      |\n\n"
      ],
      "image": "vlmx-ocrpage_2.jpg",
      "width": 2068,
      "height": 842
    },
    {
      "id": "ocr-page_3",
      "task": "ocr-page",
      "metric": "ocr",
      "prompt": "Transcribe all the text in this document as Markdown. Output only the transcription.",
      "gold": [
        "## Initial Health Screening\n\nFirst Name: Matthew\n\nLast Name: Brown\n\nMarital Status: Single □ Married ☑ Divorced □ Separated □ Widowed □ \n\nAddress: 4322 Autumn Terrace\n\nCity: Limestone\n\nState: PA\n\nZip: 16234\n\nPhone: 809-916-9601\n\nDOB: December 3, 1953\n\nSpouse Name: John Brown\n\nSpouse Phone: 550 605-4208\n\nIs the requested medication NEW □  or a CONTINUATION ☑  of THERAPY? If so, start date: \\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\n\nHave you been hospitalized in the past year? Yes ☑  No □ \n\n#### Social History\n\nDo you smoke? Yes □  No ☑ \n\nIf yes, how many times per week: 1 ○ 2 ○ 3 ○ 4 ○ 5 ○ 6 ○ 7 ○ 8 ○ 9 ○ 10+ ○ \n\nDo you drink? Yes □  No ☑ \n\nIf yes, how many times per week: 1 ○ 2 ○ 3 ○ 4 ○ 5 ○ 6 ○ 7 ○ 8 ○ 9 ○ 10+ ○ \n\nDo you exercise? Yes ☑  No □ \n\nIf yes, how many times per week: 1 ● 2 ○ 3 ○ 4 ○ 5 ○ 6 ○ 7 ○ 8 ○ 9 ○ 10+ ○ \n\nDo you have a social support system? Yes ☑  No □ \n\n#### Family Health History\n\n| **Relation** | **Age If Living** | **Age At Death** | **Chronic Health Problems** |\n| ------------ | ----------------- | ---------------- | --------------------------- |\n| Father       | 54                |                  | osteoporosis                |\n| Mother       | 56                |                  | hypertension                |\n| Sister       | 14                |                  | cancer, hearing loss        |\n"
      ],
      "image": "vlmx-ocrpage_3.jpg",
      "width": 2303,
      "height": 1738
    },
    {
      "id": "ocr-page_4",
      "task": "ocr-page",
      "metric": "ocr",
      "prompt": "Transcribe all the text in this document as Markdown. Output only the transcription.",
      "gold": [
        "# SaveMor\n\nBronberg SaveMor\nTel: (012) 817 2119\nVat Reg: 4410166088\n\nGOLDI FROZEN CHICKEN NE 1KG 39.99 A\nJABA S/PWD CURRY 15GR \n  2 @ 1.49     2.98 A\nSAVEMOR CARRIER 1'S 1.16 A\nTOTAL FOR 4 ITEMS 44.13\nTENDERED Cash 50.00\nCHANGE Cash 5.90\nROUNDING 0.03\nROUNDED TOTAL 44.10\n\nTAX INVOICE\nVAT rate excl. TAX incl.\n15.00% 38.37 5.76 44.13 A\n\nSLIP / TILL / CASHIER / DATE / TIME\n9958 004 104 04.04.24 15:51\nCASHIER NAME: POS 4\n\nThank you for your support!\nPlease call again"
      ],
      "image": "vlmx-ocrpage_4.jpg",
      "width": 1080,
      "height": 1920
    },
    {
      "id": "ocr-small_0",
      "task": "ocr-small",
      "metric": "ocr",
      "prompt": "Read the text in the image. Output only the text, nothing else.",
      "gold": [
        "A bad workman always blames his tools."
      ],
      "image": "vlmx-ocrsmall_0.jpg",
      "width": 800,
      "height": 800
    },
    {
      "id": "ocr-small_1",
      "task": "ocr-small",
      "metric": "ocr",
      "prompt": "Read the text in the image. Output only the text, nothing else.",
      "gold": [
        "Honesty is the best policy."
      ],
      "image": "vlmx-ocrsmall_1.jpg",
      "width": 800,
      "height": 534
    },
    {
      "id": "ocr-small_2",
      "task": "ocr-small",
      "metric": "ocr",
      "prompt": "Read the text in the image. Output only the text, nothing else.",
      "gold": [
        "What is done cannot be undone"
      ],
      "image": "vlmx-ocrsmall_2.jpg",
      "width": 984,
      "height": 224
    },
    {
      "id": "ocr-small_3",
      "task": "ocr-small",
      "metric": "ocr",
      "prompt": "Read the text in the image. Output only the text, nothing else.",
      "gold": [
        "A tree is known by its fruit"
      ],
      "image": "vlmx-ocrsmall_3.jpg",
      "width": 354,
      "height": 258
    },
    {
      "id": "ocr-small_4",
      "task": "ocr-small",
      "metric": "ocr",
      "prompt": "Read the text in the image. Output only the text, nothing else.",
      "gold": [
        "The pillow is the best advisor"
      ],
      "image": "vlmx-ocrsmall_4.jpg",
      "width": 700,
      "height": 368
    },
    {
      "id": "tiling-perf_0",
      "task": "tiling-perf",
      "metric": "vqa",
      "prompt": "What fruits do you see on the plate? List each one you can identify.",
      "gold": [
        "apple, plum, raspberry, blackberry, blueberry"
      ],
      "image": "vlmx-tilingperf_0.jpg",
      "width": 480,
      "height": 640
    },
    {
      "id": "tiling-perf_1",
      "task": "tiling-perf",
      "metric": "vqa",
      "prompt": "What fruits do you see on the plate? List each one you can identify.",
      "gold": [
        "apple, plum, raspberry, blackberry, blueberry"
      ],
      "image": "vlmx-tilingperf_1.jpg",
      "width": 1200,
      "height": 1600
    },
    {
      "id": "tiling-perf_2",
      "task": "tiling-perf",
      "metric": "vqa",
      "prompt": "What fruits do you see on the plate? List each one you can identify.",
      "gold": [
        "apple, plum, raspberry, blackberry, blueberry"
      ],
      "image": "vlmx-tilingperf_2.jpg",
      "width": 1536,
      "height": 2048
    },
    {
      "id": "tiling-perf_3",
      "task": "tiling-perf",
      "metric": "vqa",
      "prompt": "What fruits do you see on the plate? List each one you can identify.",
      "gold": [
        "apple, plum, raspberry, blackberry, blueberry"
      ],
      "image": "vlmx-tilingperf_3.jpg",
      "width": 2250,
      "height": 3000
    },
    {
      "id": "tiling-perf_4",
      "task": "tiling-perf",
      "metric": "vqa",
      "prompt": "What is the headline of this newspaper article? Give the most prominent words.",
      "gold": [
        "Titanic sinks four hours after hitting iceberg"
      ],
      "image": "vlmx-tilingperf_4.jpg",
      "width": 500,
      "height": 350
    },
    {
      "id": "tiling-perf_5",
      "task": "tiling-perf",
      "metric": "vqa",
      "prompt": "What is the headline of this newspaper article? Give the most prominent words.",
      "gold": [
        "Titanic sinks four hours after hitting iceberg"
      ],
      "image": "vlmx-tilingperf_5.jpg",
      "width": 1200,
      "height": 840
    },
    {
      "id": "tiling-perf_6",
      "task": "tiling-perf",
      "metric": "vqa",
      "prompt": "What is the headline of this newspaper article? Give the most prominent words.",
      "gold": [
        "Titanic sinks four hours after hitting iceberg"
      ],
      "image": "vlmx-tilingperf_6.jpg",
      "width": 1600,
      "height": 1120
    },
    {
      "id": "tiling-perf_7",
      "task": "tiling-perf",
      "metric": "vqa",
      "prompt": "What is the headline of this newspaper article? Give the most prominent words.",
      "gold": [
        "Titanic sinks four hours after hitting iceberg"
      ],
      "image": "vlmx-tilingperf_7.jpg",
      "width": 3200,
      "height": 2240
    },
    {"id": "disc-ocr_0", "task": "disc-ocr", "metric": "ocr", "prompt": "Transcribe all the text in this document as Markdown. Output only the transcription.", "gold": ["# Company structure\n\nBD is structured to serve customers by providing unique solutions. The data below represents the company structure for FY 2019.\n\n## Revenue by geography\n\n(millions of dollars)\n\nUnited States (including Puerto Rico)\n$9,730\n\nEurope\n$3,359\n\nGreater Asia (including Japan and Asia Pacific)\n$2,726\n\nOther (including Latin America, Canada and EMA [which includes the Commonwealth of Independent States, Middle East and Africa])\n$1,476\n\n## Revenue by segment\n\n(billions of dollars)\n\n$9.1\nBD Medical\n\n$4.3\nBD Life Sciences\n\n$3.9\nBD Interventional\n\n<table>\n    <tr>\n        <td colspan=\"2\" style=\"text-align:center;\">$17.3</td>\n    </tr>\n    <tr>\n        <td colspan=\"2\" style=\"text-align:center;\">Total BD revenue</td>\n    </tr>\n    <tr>\n        <td>Diabetes Care</td>\n        <td>$1.1</td>\n    </tr>\n    <tr>\n        <td>Medication Management Solutions</td>\n        <td>$2.6</td>\n    </tr>\n    <tr>\n        <td>Medication Delivery Solutions</td>\n        <td>$3.9</td>\n    </tr>\n    <tr>\n        <td>Urology and Critical Care</td>\n        <td>$1.1</td>\n    </tr>\n    <tr>\n        <td>Peripheral Intervention</td>\n        <td>$1.4</td>\n    </tr>\n    <tr>\n        <td>Surgery</td>\n        <td>$1.4</td>\n    </tr>\n    <tr>\n        <td>Biosciences</td>\n        <td>$1.2</td>\n    </tr>\n    <tr>\n        <td>Diagnostic Systems</td>\n        <td>$1.5</td>\n    </tr>\n    <tr>\n        <td>Preanalytical Systems</td>\n        <td>$1.6</td>\n    </tr>\n    <tr>\n        <td>Pharmaceutical Systems</td>\n        <td>$1.5</td>\n    </tr>\n</table>\n\nValues in this exhibit reflect rounded numbers in billions and include Bard.\n\nGRI disclosures: 102-6, 102-7\n<page_number>6</page_number>\n"], "image": "vlmx-discocr_0.jpg", "width": 1400, "height": 1798},
    {"id": "disc-ocr_1", "task": "disc-ocr", "metric": "ocr", "prompt": "Transcribe all the text in this document as Markdown. Output only the transcription.", "gold": ["## Medical Equipment Inspection Checklist\n\nEquipment ID: HO7CYSCQ\n\nModel: Parisian - Kuvalis 1qck\n\nDepartment: ICU\n\nInspection Date: 2025-02-13\n\nInspector: Margie Rohan\n\n### Physical Condition\n\n| Checkpoint                  | Pass | Fail | N/A | Notes |\n| --------------------------- | ---- | ---- | --- | ----- |\n| Equipment housing intact    | ✅   |      |     |       |\n| Display screen clear        | ✅   |      |     |       |\n| Control buttons functional  | ✅   |      |     |       |\n| Power cord undamaged        |      |      | -   |       |\n| Mounting brackets secure    | ✅   |      |     |       |\n| Wheels/casters condition    | ✅   |      |     |       |\n| Labels and markings legible | ✅   |      |     |       |\n| Accessories complete        | ✅   |      |     |       |\n\n### Electrical Safety\n\n| Checkpoint                | Pass | Fail | N/A | Notes |\n| ------------------------- | ---- | ---- | --- | ----- |\n| Ground connection         | ✅   |      |     |       |\n| Insulation resistance     |      | ❌   |     |       |\n| Leakage current           | ✅   |      |     |       |\n| Power supply stable       | ✅   |      |     |       |\n| Battery backup functional | ✅   |      |     |       |\n| Circuit breaker operation | ✅   |      |     |       |\n| Surge protection          |      |      | -   |       |\n| Cable connections secure  | ✅   |      |     |       |\n\n### Performance Tests\n\n| Checkpoint             | Pass | Fail | N/A | Notes                                |\n| ---------------------- | ---- | ---- | --- | ------------------------------------ |\n| Self-test passed       | ✅   |      |     |                                      |\n| Calibration check      | ✅   |      |     |                                      |\n| Alarm systems          | ✅   |      |     |                                      |\n| Backup systems         | ✅   |      |     |                                      |\n| Output accuracy        |      |      | -   | Not required for this equipment type |\n| Response time          | ✅   |      |     |                                      |\n| Data storage/retrieval |      | ❌   |     |                                      |\n| Network connectivity   | ✅   |      |     |                                      |\n| Sensor accuracy        | ✅   |      |     |                                      |\n\n### Overall Status\n\nInspection Result: ❌\n\nNext Inspection Due: 2025-06-22\n"], "image": "vlmx-discocr_1.jpg", "width": 1400, "height": 1848},
    {"id": "disc-ocr_2", "task": "disc-ocr", "metric": "ocr", "prompt": "Transcribe all the text in this document as Markdown. Output only the transcription.", "gold": ["![Chase watermark](image)\n\nElta Kuhlman-Rowe25008 Sebastian StreamLeesburg, Montana 18159\n\n![Chase](image)\n\n2025-02-03\n\nPAY TO THE ORDER OF\n\nCronin and Sons\n\n$13,164.60\n\nThirteen Thousand One Hundred Sixty Four and 60/100\n\nDOLLARS\n\n⑆712795623⑆  3875146359  ⑈3280⑈\n\nChristelle Morar375 Clarence StreetNew Jordynborough, California 16443-6854\n\n![Chase](image)\n\n2025-02-03\n\nPAY TO THE ORDER OF\n\nSauer - Bergstrom\n\n$8,058.51\n\nEIGHT THOUSAND FIFTY EIGHT and 51/100\n\nDOLLARS\n\nMEMO\n\nUtility payment\n\n⑆778891305⑆  6850532756  ⑈6956⑈\n\nSean Rodriguez14959 Goldner HarborsDavie, North Carolina 11933\n\n![Wells Fargo](image)\n\n2025-02-03\n\nPAY TO THE ORDER OF\n\nJast Inc\n\n$2,607.92\n\nTwo Thousand Six Hundred Seven and 92/100\n\nDOLLARS\n\n⑆898078646⑆  7872335959  ⑈1485⑈\n\nGerhard Pacocha58864 W Center StreetHellerburgh, Florida 16539-8897\n\n![U.S. Bank](image)\n\n2025-02-03\n\nPAY TO THE ORDER OF\n\nWaelchi and Sons\n\n$466.92\n\nFOUR HUNDRED SIXTY SIX and 92/100\n\nDOLLARS\n\nGerhard Pacocha\n\n⑆395668369⑆  8769160829  ⑈6741⑈\n\nLora Anderson462 Bailee HarborSouth Jordon, Montana 35569\n\n![Chase](image)\n\n2025-02-03\n\nPAY TO THE ORDER OF\n\nLarson, Reichel and Olson\n\n$46.80\n\nForty Six and 80/100\n\nDOLLARS\n\nMEMO\n\nContract payment\n\nLora Anderson\n\n⑆204019515⑆  1146715701  ⑈7961⑈\n\n![Wells Fargo watermark](image)\n\nWinston Brakus8715 N Cedar StreetPort Freeda, Florida 23892-9588\n\n![Wells Fargo](image)\n\n2025-02-03\n\nPAY TO THE ORDER OF\n\nPredovic - Auer\n\n$474.73\n\nFour Hundred Seventy Four and 73/100\n\nDOLLARS\n\n⑆885864271⑆  9311721185  ⑈9163⑈\n"], "image": "vlmx-discocr_2.jpg", "width": 1768, "height": 1400},
    {"id": "disc-ocr_3", "task": "disc-ocr", "metric": "ocr", "prompt": "Transcribe all the text in this document as Markdown. Output only the transcription.", "gold": ["# Transaction Overview by Quarter\n\n| City          | Year | Transactions (Total) | Permits Filed | Educational Buildings | Data Centers | Healthcare Facilities | State & Federal Contruction |\n| ------------- | ---- | -------------------- | ------------- | --------------------- | ------------ | --------------------- | --------------------------- |\n| Phoenix       | 2021 | $208M                | 132           | $72.1M                | $32.8M       | $22.9M                | $80.3M                      |\n| Charlotte     | 2021 | $181M                | 469           | $34.8M                | $95.1M       | $29.7M                | $21.5M                      |\n| San Diego     | 2021 | $188M                | 347           | $72.3M                | $17.7M       | $57.8M                | $40.2M                      |\n| San Francisco | 2021 | $46M                 | 76            | $18.4M                | $6.2M        | $14.2M                | $7.1M                       |\n\n### Educational Buildings\n\nAvg: $9.8M\n\nMedian: $10.5M\n\n### Data Centers\n\nAvg: $28.0M\n\nMedian: $28.5M\n\n### Healthcare Facilities\n\nAvg: $22.0M\n\nMedian: $26.0M\n\n### State & Federal Contruction\n\nAvg: $15.0M\n\nMedian: $15.5M\n\n### Transactions by Quarter\n\n!\\[Bar chart showing Transactions by Quarter\\]\n\n| date    | Transaction Amount ($M) | category                    |\n| ------- | ----------------------- | --------------------------- |\n| Q1-2021 | 9                       | Educational Buildings       |\n| Q2-2021 | 12                      | Educational Buildings       |\n| Q3-2021 | 12                      | Educational Buildings       |\n| Q4-2021 | 6                       | Educational Buildings       |\n| Q1-2021 | 40                      | Data Centers                |\n| Q2-2021 | 19                      | Data Centers                |\n| Q3-2021 | 38                      | Data Centers                |\n| Q4-2021 | 15                      | Data Centers                |\n| Q1-2021 | 25                      | Healthcare Facilities       |\n| Q2-2021 | 27                      | Healthcare Facilities       |\n| Q3-2021 | 28                      | Healthcare Facilities       |\n| Q4-2021 | 8                       | Healthcare Facilities       |\n| Q1-2021 | 8                       | State & Federal Contruction |\n| Q2-2021 | 21                      | State & Federal Contruction |\n| Q3-2021 | 18                      | State & Federal Contruction |\n| Q4-2021 | 13                      | State & Federal Contruction |\n\n![Costar Logo](image)\n\nCostar Investment Opportunities\n\nPage 13\n"], "image": "vlmx-discocr_3.jpg", "width": 1566, "height": 1400},
    {"id": "disc-ocr_4", "task": "disc-ocr", "metric": "ocr", "prompt": "Transcribe all the text in this document as Markdown. Output only the transcription.", "gold": ["# CALIFORNIA COMMERCIAL LEASE AGREEMENT\n\nThis Lease Agreement made the\n\n2025-05-11\n\n, by and between\n\nJackie Gerhold\n\n\\[name of lessor\\], of\n\n653 Woodside Road\n\n\\[street address\\], State of\n\nCalifornia\n\n, hereinafter referred to as \"Lessor\", and\n\nLemke Inc\n\n\\[name of lessee\\], of\n\n152 Letitia Landing\n\n\\[street address\\], State of\n\nCalifornia\n\n, hereinafter referred to as \"Lessee\", collectively referred to herein as the \"Parties\", agree as follows:\n\n## 1\\. DESCRIPTION OF LEASED PREMISES:\n\nThe Lessor agrees to lease to the Lessee the following described\n\n9388\n\nsquare feet (SF) of\n\nOffice Space\n\nlocated at\n\n17803 Gorczany Throughway\n\n\\[street address\\], State of California.\n\n## 2\\. TERM OF LEASE:\n\nThe term of this Lease shall be for a period of\n\n2\n\nyear(s)\n\n10\n\nmonth(s) commencing on the\n\n11\n\nday of\n\nMay 2025\n\nday of\n\nMay 2025\n\n, and expiring at Midnight on the\n\n11\n\nday of\n\nMarch 2028\n\nday of\n\nMarch 2028\n\n. (\"Initial Term\")\n\n## 3\\. BASE RENT:\n\nThe net monthly payment shall be\n\n3400\n\ndollars ($3400) per month. A security deposit of\n\n2089\n\ndollars ($2089) is required. payable monthly with the first payment due upon the is required. payable monthly with the first payment due upon the commencement of the Lease and each monthly installment payable thereafter on the 1st day of each month\n\n## 4\\. Options to Renew (Check one):\n\n ☑Lessee may not renew the Lease. ☑The Lessee may have the right to renew the Lease with a total of \\_\\_\\_\\_ renewal period(s) with each term being \\_\\_\\_\\_ year(s) \\_\\_\\_\\_ month(s) which may be exercised by giving written notice to Lessor no less than 60 days prior to the expiration of the Lease or renewal period.\n\nPrepared using zipForm® software\n\nPage 5 of 8\n"], "image": "vlmx-discocr_4.jpg", "width": 1400, "height": 1550},
    {"id": "disc-ocr_5", "task": "disc-ocr", "metric": "ocr", "prompt": "Transcribe all the text in this document as Markdown. Output only the transcription.", "gold": ["[uptime intelligence logo]\nKEYNOTE REPORT\nUptime Institute Global Data Center Survey 2024\n\nFigure 1\nCost issues are the top concern for management in 2024\nLooking at the next 12 months, how concerned is your digital infrastructure management regarding each of the following issues? (n=638)\n\n<table>\n  <tr>\n    <th></th>\n    <th>Very concerned</th>\n    <th>Somewhat concerned</th> \n    <th>Slightly concerned</th>\n    <th>Not concerned</th>\n  </tr>\n  <tr>\n    <td>Cost</td>\n    <td>44%</td>\n    <td>36%</td>\n    <td>17%</td>\n    <td>3%</td>\n  </tr>\n  <tr>\n    <td>Forecasting future data center capacity requirements</td>\n    <td>30%</td>\n    <td>38%</td>\n    <td>24%</td>\n    <td>9%</td>\n  </tr>\n  <tr>\n    <td>Improving energy performance for facilities equipment</td>\n    <td>29%</td>\n    <td>40%</td>\n    <td>23%</td>\n    <td>8%</td>\n  </tr>\n  <tr>\n    <td>Lack of qualified staff</td>\n    <td>27%</td>\n    <td>44%</td>\n    <td>24%</td>\n    <td>6%</td>\n  </tr>\n  <tr>\n    <td>Accommodating significantly denser IT</td>\n    <td>24%</td>\n    <td>41%</td>\n    <td>25%</td>\n    <td>9%</td>\n  </tr>\n  <tr>\n    <td>Improving energy performance for IT</td>\n    <td>24%</td>\n    <td>40%</td>\n    <td>26%</td>\n    <td>10%</td>\n  </tr>\n</table>\n\nUPTIME INSTITUTE GLOBAL SURVEY OF IT AND DATA CENTER MANAGERS 2024\n[uptime intelligence logo]\n\nAlthough AI and high-performance compute workloads garner significant media attention, their broader industry impact will likely take time to materialize. Managers also report that the need to accommodate higher density workloads is a less pressing concern than rising costs and the need for greater energy efficiency. While 65% of operators are at least somewhat concerned about their ability to accommodate rising densities, most are currently planning to meet the need for denser IT infrastructure by limiting their efforts to specific areas of their data halls.\n\nCompared with 2023, other response categories show little change. About two-thirds of managers are at least somewhat concerned with forecasting future data center capacity requirements, a lack of qualified staff, and energy performance for IT and facilities equipment.\n"], "image": "vlmx-discocr_5.jpg", "width": 1400, "height": 1766},
    {"id": "disc-ocr_6", "task": "disc-ocr", "metric": "ocr", "prompt": "Transcribe all the text in this document as Markdown. Output only the transcription.", "gold": ["# Real Estate Transaction Highlights\n\n| City          | Year | Transactions (Total) | Permits Filed | Healthcare Facilities | Commercial Industrial | Retail | Data Centers |\n| ------------- | ---- | -------------------- | ------------- | --------------------- | --------------------- | ------ | ------------ |\n| San Francisco | 2024 | $48M                 | 269           | $6.5M                 | $4.0M                 | $24.5M | $12.9M       |\n| New York      | 2024 | $271M                | 431           | $67.0M                | $72.2M                | $54.4M | $77.4M       |\n| Philadelphia  | 2024 | $213M                | 95            | $16.1M                | $73.4M                | $54.6M | $68.9M       |\n| Denver        | 2024 | $269M                | 362           | $80.9M                | $39.1M                | $64.5M | $84.5M       |\n| Phoenix       | 2024 | $234M                | 493           | $75.2M                | $50.1M                | $80.2M | $28.4M       |\n| Indianapolis  | 2024 | $123M                | 359           | $46.9M                | $22.0M                | $45.4M | $8.8M        |\n\n### Healthcare Facilities\n\nAvg: $23.8M\n\nMedian: $24.0M\n\n### Commercial Industrial\n\nAvg: $22.8M\n\nMedian: $26.0M\n\n### Retail\n\nAvg: $2.8M\n\nMedian: $2.0M\n\n### Data Centers\n\nAvg: $19.3M\n\nMedian: $16.5M\n\n### Transactions by Quarter\n\n!\\[Bar chart showing Transactions by Quarter\\]\n\n| date    | Transaction Amount ($M) | category              |\n| ------- | ----------------------- | --------------------- |\n| Q1-2024 | 17                      | Healthcare Facilities |\n| Q2-2024 | 32                      | Healthcare Facilities |\n| Q3-2024 | 31                      | Healthcare Facilities |\n| Q4-2024 | 15                      | Healthcare Facilities |\n| Q1-2024 | 27                      | Commercial Industrial |\n| Q2-2024 | 25                      | Commercial Industrial |\n| Q3-2024 | 6                       | Commercial Industrial |\n| Q4-2024 | 33                      | Commercial Industrial |\n| Q1-2024 | 2                       | Retail                |\n| Q2-2024 | 5                       | Retail                |\n| Q3-2024 | 2                       | Retail                |\n| Q4-2024 | 2                       | Retail                |\n| Q1-2024 | 36                      | Data Centers          |\n| Q2-2024 | 19                      | Data Centers          |\n| Q3-2024 | 14                      | Data Centers          |\n| Q4-2024 | 8                       | Data Centers          |\n\nRealty Income (The Monthly Dividend Company) Investment Opportunities\n\nPage 15 of 33\n"], "image": "vlmx-discocr_6.jpg", "width": 1566, "height": 1400},
    {"id": "disc-ocr_7", "task": "disc-ocr", "metric": "ocr", "prompt": "Transcribe all the text in this document as Markdown. Output only the transcription.", "gold": ["**Advertisement**\n\n## CRUISE WITH SCIENTIFIC AMERICAN Travel\n\n### Celebrate Scientific American’s 175th Anniversary\n\nCruise the Pacific Rim of South and Central America to celebrate **Scientific American’s 175th Anniversary**. Savor 20+ hours of exclusive onboard classes while we’re at sea. While we’re in port, take advantage of archaeology, fitness, food, history, and outdoor opportunities.\n\nSurvey the big history of the region. Get a cosmic perspective on the search for life in the universe. Enrich your knowledge of regional pre-Columbian peoples. And then head ashore and deepen your appreciation of the cultures and beauty of the area.\n\nJoin us! Hail the spirit of inquiry, the discipline of scientific theory, and the value of fact on **Scientific American’s** 175th birthday cruise. Get in on the action and book now.\n\n### THE AMERICAS, MARCH 15th – 30th, 2020\n\n- **SAN DIEGO**\n- Cabo San Lucas\n- Huatulco\n- Puerto Vallarta\n- Puntarenas\n- Manta\n- Lima\n- La Serena (Coquimbo)\n- **SAN ANTONIO**\n\n### SPEAKERS & SEMINARS\n\nThe conference fee is $1,575 and includes all 90-minute seminars below.\n\n#### Ken Albala, Ph.D.\n**Professor of History, University of the Pacific**\n\nKen Albala teaches food history and the history of medicine. He has written 25 books and is the creator of the Great Courses’ “Food: A Cultural Culinary History” and other video courses. He has written numerous books, including cookbooks, popular histories, encyclopedias and reference works, winning awards for **Beans: A History and Three World Cuisines**. He has been a visiting scholar at the University of the Pacific, Boston University, and the University of Leeds.\n\n- **Anthropology: Revelations of Cookbooks**\n  - Gastronomy in the Ancient World\n  - The Medieval Culinary Aesthetic from Baghdad to Paris\n  - The Renaissance Kitchen\n  - Cookbooks for Mass Consumption\n\n#### David Christian, Ph.D.\n**Distinguished Professor, Modern History, Macquarie University**\n\nDavid Christian began teaching courses in Big History in the 1980s and has been at the forefront of many educational Big History projects, including the free online Big History Project, which Bill Gates, and Macquarie University’s Big History Institute are also creating the Big History School for K-12 online courses.\n\n- **Big History: A “Short” History of the Universe and Everything**\n  - The Cosmos\n  - A Living Planet\n  - Life\n  - The Future: Where Is It All Going?\n\n#### Robert Hazen, Ph.D.\n**Clarence Robinson Professor of Earth Sciences, George Mason University**\n\nRobert Hazen is also the Senior Staff Scientist at the Carnegie Institution’s Geophysical Laboratory. He is the Executive Director of the Deep Carbon Observatory, where his research focuses on the role of minerals in the origin of life and the co-evolution of the geo- and biospheres, between biomolecules and minerals.\n\n- **Geology: Minerals and the Origins of Life**\n  - How Rocks and Life Co-evolved\n  - Mysteries of the Evolving Mineral Realm\n  - Carbon and the Emergence of (Almost) Everything\n  - The Scientific Quest for Life’s Origins\n\nCruise prices start at $2,439 per person (pp) (based on double occupancy). Add (1) fees, taxes, and gratuities apply. Cruise pricing is subject to change.\n\n**For more info email:** Info@InsightCruises.com | **or visit:** ScientificAmerican.com/AnniversaryCruise\n<hr><hr>\n"], "image": "vlmx-discocr_7.jpg", "width": 1400, "height": 1849},
    {"id": "disc-ocr_8", "task": "disc-ocr", "metric": "ocr", "prompt": "Transcribe all the text in this document as Markdown. Output only the transcription.", "gold": ["# CALIFORNIA COMMERCIAL LEASE AGREEMENT\n\nThis Lease Agreement made the\n\n2025-06-01\n\n, by and between\n\nDoyle Grant\n\n\\[name of lessor\\], of\n\n3224 Brennon Street\n\n\\[street address\\], State of\n\nCalifornia\n\n, hereinafter referred to as \"Lessor\", and\n\nHagenes Inc\n\n\\[name of lessee\\], of\n\n35685 Vergie Hollow\n\n\\[street address\\], State of\n\nCalifornia\n\n, hereinafter referred to as \"Lessee\", collectively referred to herein as the \"Parties\", agree as follows:\n\n## 1\\. DESCRIPTION OF LEASED PREMISES:\n\nThe Lessor agrees to lease to the Lessee the following described\n\n7480\n\nsquare feet (SF) of\n\nWarehouse\n\nlocated at\n\n59958 3rd Avenue\n\n\\[street address\\], State of California.\n\n## 2\\. TERM OF LEASE:\n\nThe term of this Lease shall be for a period of\n\n1\n\nyear(s)\n\n0\n\nmonth(s) commencing on the\n\n1\n\nday of\n\nJune 2025\n\nday of\n\nJune 2025\n\n, and expiring at Midnight on the\n\n1\n\nday of\n\nJune 2026\n\nday of\n\nJune 2026\n\n. (\"Initial Term\")\n\n## 3\\. BASE RENT:\n\nThe net monthly payment shall be\n\n9993\n\ndollars ($9993) per month. A security deposit of\n\n5324\n\ndollars ($5324) is required. payable monthly with the first payment due upon the is required. payable monthly with the first payment due upon the commencement of the Lease and each monthly installment payable thereafter on the 1st day of each month\n\n## 4\\. Options to Renew (Check one):\n\n ☑Lessee may not renew the Lease. ☑The Lessee may have the right to renew the Lease with a total of \\_\\_\\_\\_ renewal period(s) with each term being \\_\\_\\_\\_ year(s) \\_\\_\\_\\_ month(s) which may be exercised by giving written notice to Lessor no less than 60 days prior to the expiration of the Lease or renewal period.\n\nPrepared using zipForm® software\n\nPage 5 of 8\n"], "image": "vlmx-discocr_8.jpg", "width": 1400, "height": 1542}
  ]
}
