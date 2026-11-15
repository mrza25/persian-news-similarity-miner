# Persian & English News Similarity Miner — سامانه دوزبانه داده‌کاوی سنجش شباهت متون خبری

پروژه درس **داده‌کاوی (Data Mining) کارشناسی ارشد** — طراحی و پیاده‌سازی پایپ‌لاین متن‌کاوی دوزبانه (**فارسی و انگلیسی**)، مدل فضای برداری (`Sublinear TF-IDF`)، ضرایب مجموعه‌ای (`Jaccard`، `Sørensen–Dice`، `Overlap`)، الگوریتم **MinHash و Locality-Sensitive Hashing (LSH)**، وب‌اسکریپر سایت‌های خبری فارسی و انگلیسی و **هم‌ترازی معنایی جملات (`Cross-Document Sentence Alignment`)**.

---

## ۱. ویژگی‌های کلیدی پروژه

1. **پشتیبانی کامل از سایت‌ها و متون خبری انگلیسی و فارسی (`Bilingual FA & EN`)**:
   - **وب‌اسکریپر ۴ لایه (`fetch_news_from_url`)**: استخراج خودکار بدنه خبر از ساختار `JSON-LD (articleBody)` (در سایت‌هایی مانند **The Guardian**، **BBC**، **Reuters**، **NYT**)، تگ‌های معنایی `<article>` و `<main>`، تگ‌های `<p>` و متاتگ‌های `OpenGraph`.
   - **تشخیص خودکار زبان (`detect_language`)**: تشخیص متن فارسی یا انگلیسی و اعمال خودکار ایست‌واژه‌های دوزبانه (`ALL_STOPWORDS`).
   - **ریشه‌یاب و لماتایزر انگلیسی (`stem_english_word`) + ریشه‌یاب محافظت‌شده فارسی (`stem_word`)**: یکسان‌سازی املایی بریتانیایی و آمریکایی (`levelled/equalised -> equalized`، `seventh -> 7th`)، تبدیل افعال بی‌قاعده انگلیسی (`came -> come`, `drew -> draw`)، حذف پسوندهای انگلیسی (`-ing`, `-ed`, `-ies`, `-es`, `-s`) و ریشه‌یابی فارسی با حفظ اسامی خاص (`PROTECTED_WORDS`).
2. **حالت‌های پیش‌فرض اخبار ورزشی فوتبالی (فارسی و انگلیسی)**:
   - **پیش‌فرض فارسی ۱ (`irna_vs_mehr`)**: مقایسه گزارش مسابقه تیم ملی فوتبال ایران و فلسطین در تورنمنت بصره (۲۳ آبان ۱۴۰۵ / ۱۴ نوامبر ۲۰۲۶) از دو منبع [خبرگزاری جمهوری اسلامی (ایرنا)](https://www.irna.ir/news/86321699/%D8%A7%DB%8C%D8%B1%D8%A7%D9%86-%DB%B2-%D9%81%D9%84%D8%B3%D8%B7%DB%8C%D9%86-%D8%B5%D9%81%D8%B1-%D8%AC%D9%88%D8%A7%D9%86-%D9%87%D8%A7-%D8%A7%D9%85%DB%8C%D8%AF%D9%88%D8%A7%D8%B1%D9%85%D8%A7%D9%86-%DA%A9%D8%B1%D8%AF%D9%86%D8%AF) و [خبرگزاری مهر](https://www.mehrnews.com/news/7010871/%DA%A9%D8%A7%D8%B1%D8%AA-%D9%82%D8%B1%D9%85%D8%B2-%D8%A8%D9%87-%D8%AF%D8%A7%D8%AF-%D9%82%D9%84%D8%B9%D9%87-%D9%86%D9%88%DB%8C%DB%8C-%D8%B1%D8%B3%DB%8C%D8%AF-%D9%BE%DB%8C%D8%B1%D9%88%D8%B2%DB%8C-%D8%B3%D8%AE%D8%AA-%D8%A7%DB%8C%D8%B1%D8%A7%D9%86-%D8%A8%D8%B1%D8%A7%D8%A8%D8%B1-%D9%81%D9%84%D8%B3%D8%B7%DB%8C%D9%86).
   - **پیش‌فرض انگلیسی (`english_guardian_vs_reuters`)**: مقایسه گزارش مسابقه ایران ۲ - ۲ نیوزیلند در جام جهانی ۲۰۲۶ از دو منبع بین‌المللی [The Guardian](https://www.theguardian.com/football/2026/jun/16/iran-new-zealand-world-cup-2026-group-g-match-report) و [Arab News / Reuters](https://www.arabnews.com/football/iran-fight-back-to-draw-2-2-with-new-zealand-at-world-cup-2647332).
   - **پیش‌فرض فارسی ۲ (`isna_vs_football360`)**: مقایسه گزارش بازی ایران و فلسطین از [ایسنا](https://www.isna.ir/news/1405082315926/%D8%A8%D8%B1%D8%AF-%DB%B2-%DA%AF%D9%84%D9%87-%D8%B4%D8%A7%DA%AF%D8%B1%D8%AF%D8%A7%D9%86-%D9%82%D9%84%D8%B9%D9%87-%D9%86%D9%88%DB%8C%DB%8C-%D8%A8%D8%B1%D8%A7%D8%A8%D8%B1-%D9%81%D9%84%D8%B3%D8%B7%DB%8C%D9%86-%DB%B1%DB%B0-%D9%86%D9%81%D8%B1%D9%87) و [فوتبال ۳۶۰](https://football360.ir/post/20261114127/%D8%A7%DB%8C%D8%B1%D8%A7%D9%86-2-0-%D9%81%D9%84%D8%B3%D8%B7%DB%8C%D9%86-%D8%AF%D8%B1%D8%AE%D8%B4%D8%B4-%D8%AC%D9%88%D8%A7%D9%86%D9%87%D8%A7%DB%8C-%D8%A7%D9%85%DB%8C%D8%B1-%D8%AF%D8%B1-%D8%A8%D8%B5%D8%B1%D9%87).
   - **کنترل منفی (`different_football_news`)**: مقایسه بازی ایران و فلسطین با خبر [خداحافظی سامان قدوس از تیم ملی در ایرنا](https://www.irna.ir/news/86317660/%D8%AA%D8%B5%D9%85%DB%8C%D9%85-%D8%B9%D8%AC%DB%8C%D8%A8-%D8%B3%D8%AA%D8%A7%D8%B1%D9%87-%D8%AA%DB%8C%D9%85-%D9%85%D9%84%DB%8C-%D9%81%D9%88%D8%AA%D8%A8%D8%A7%D9%84-%D9%82%D8%AF%D9%88%D8%B3-%D8%A7%D8%B2-%D8%AA%DB%8C%D9%85-%D9%85%D9%84%DB%8C-%D8%AE%D8%AF%D8%A7%D8%AD%D8%A7%D9%81%D8%B8%DB%8C-%DA%A9%D8%B1%D8%AF).

---

## ۲. خلاصه نتایج تجربی روی ۴ سناریوی فارسی و انگلیسی

| معیار داده‌کاوی (Similarity Metric) | ایرنا در برابر مهر (فارسی - هم‌رویداد) | Guardian vs Reuters (انگلیسی - هم‌رویداد) | ایسنا در برابر فوتبال ۳۶۰ (فارسی - هم‌رویداد) | بازی در برابر خداحافظی قدوس (متفاوت) |
| :--- | :---: | :---: | :---: | :---: |
| **Unigram TF-IDF Cosine** | `0.5078` (50.8%) | `0.5974` (59.7%) | `0.4415` (44.1%) | `0.1164` (11.6%) |
| **1-2 gram TF-IDF Cosine** | `0.3193` (31.9%) | `0.3993` (39.9%) | `0.2666` (26.7%) | `0.0738` (7.4%) |
| **Sørensen–Dice Similarity** | `0.5591` (55.9%) | `0.6386` (63.9%) | `0.4831` (48.3%) | `0.1387` (13.9%) |
| **Word Jaccard Similarity** | `0.3880` (38.8%) | `0.4690` (46.9%) | `0.3185` (31.9%) | `0.0745` (7.5%) |
| **Overlap Coefficient** | `0.5726` (57.3%) | `0.6625` (66.2%) | `0.5119` (51.2%) | `0.2449` (24.5%) |
| **LSH Candidate Match (32×4)** | `False` (J=29.3%) | **`True` (J=38.1%)** | `False` (J=26.1%) | `False` (J=6.3%) |
| **Sentence Alignment Score** | `0.3591` (35.9%) | `0.4492` (44.9%) | `0.3595` (35.9%) | `0.1112` (11.1%) |
| **Hybrid Overall Score** | **`0.5038` (50.4%)** | **`0.5911` (59.1%)** | **`0.4496` (45.0%)** | **`0.1466` (14.7%)** |

---

## ۳. راهنمای اجرا (CLI و پنل وب)

```bash
# ۱. اجرای حالت پیش‌فرض فارسی (ایرنا و مهر) یا انگلیسی (The Guardian و Reuters)
python3 persian_news_similarity_miner.py --preset irna_vs_mehr --json-out default_result.json
python3 persian_news_similarity_miner.py --preset english_guardian_vs_reuters --json-out english_result.json

# ۲. مقایسه دو خبر از سایت‌های انگلیسی یا فارسی با دادن لینک مستقیم (URL)
python3 persian_news_similarity_miner.py \
  --url1 "https://www.theguardian.com/football/2026/jun/16/iran-new-zealand-world-cup-2026-group-g-match-report" \
  --url2 "https://www.arabnews.com/football/iran-fight-back-to-draw-2-2-with-new-zealand-at-world-cup-2647332"

# ۳. مقایسه هر دو خبر دلخواه انگلیسی یا فارسی با ارسال مستقیم متن
python3 persian_news_similarity_miner.py \
  --title1 "Match Report A" --source1 "BBC Sport" \
  --text1 "Arsenal defeated Chelsea 2-0 at Emirates Stadium thanks to two second-half goals from Bukayo Saka." \
  --title2 "Match Report B" --source2 "Sky Sports" \
  --text2 "Bukayo Saka scored twice in the second half as Arsenal beat Chelsea two-nil at the Emirates Stadium."

# ۴. اجرای ۷ تست واحد خودکار (فارسی و انگلیسی)
python3 -m unittest discover -s tests -v
``` Miner with TF-IDF, MinHash/LSH, and Sentence Alignment
