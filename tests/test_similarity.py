"""
تست‌های واحد (Unit Tests) برای راستی‌آزمایی پیش‌پردازش دوزبانه (فارسی و انگلیسی)،
بردارسازی TF-IDF، الگوریتم MinHash/LSH و مقایسه زوج خبرهای فوتبالی فارسی و انگلیسی.
"""

import unittest
from persian_news_similarity_miner import (
    DEFAULT_ARTICLES,
    NewsSimilarityEngine,
    PersianNewsPreprocessor,
    get_default_english_pair,
    get_default_pair,
)


class TestBilingualNewsSimilarity(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = PersianNewsPreprocessor()
        self.engine = NewsSimilarityEngine(self.preprocessor)

    def test_normalization_and_digits(self) -> None:
        raw = "تیمِ ملّی يوزها در دقیقه ۷۶ با ۲ گل پیروز شد!"
        norm = self.preprocessor.normalize(raw)
        self.assertIn("یوزها", norm)
        self.assertIn("76", norm)
        self.assertIn("2", norm)
        self.assertNotIn("ِ", norm)

    def test_bilingual_stemming_and_canonicalization(self) -> None:
        self.assertEqual(self.preprocessor.stem_word("بازیکنان"), "بازیکن")
        self.assertEqual(self.preprocessor.stem_word("موقعیت‌های"), "موقعیت")
        self.assertEqual(self.preprocessor.stem_word("ایران"), "ایران")
        self.assertEqual(self.preprocessor.stem_word("players"), "player")
        self.assertEqual(self.preprocessor.stem_word("came"), "come")
        self.assertEqual(self.preprocessor.stem_word("scoring"), "scor")
        norm_uk = self.preprocessor.normalize("Iran levelled in the seventh minute")
        norm_us = self.preprocessor.normalize("Iran equalized in the 7th minute")
        self.assertEqual(norm_uk, norm_us)

    def test_identical_texts_have_unit_similarity(self) -> None:
        a1, _ = get_default_pair()
        report = self.engine.compare(a1.content, a1.content)
        self.assertAlmostEqual(report.unigram_cosine_similarity, 1.0, places=5)
        self.assertAlmostEqual(report.jaccard_word_similarity, 1.0, places=5)
        self.assertTrue(report.lsh_candidate_match)

    def test_default_persian_football_pair_irna_vs_mehr(self) -> None:
        a1, a2 = get_default_pair()
        report = self.engine.compare(a1.content, a2.content)
        self.assertGreater(report.unigram_cosine_similarity, 0.45)
        self.assertGreater(report.hybrid_overall_score, 0.45)
        self.assertGreater(report.shared_unique_terms_count, 20)

    def test_default_english_football_pair_guardian_vs_reuters(self) -> None:
        en1, en2 = get_default_english_pair()
        report = self.engine.compare(en1.content, en2.content)
        self.assertEqual(self.preprocessor.detect_language(en1.content), "en")
        self.assertGreater(report.unigram_cosine_similarity, 0.55)
        self.assertGreater(report.hybrid_overall_score, 0.55)
        self.assertGreater(report.sentence_alignment_score, 0.44)

    def test_same_match_higher_than_different_football_news(self) -> None:
        irna_match = DEFAULT_ARTICLES["irna_iran_palestine"]
        mehr_match = DEFAULT_ARTICLES["mehr_iran_palestine"]
        irna_ghoddos = DEFAULT_ARTICLES["irna_ghoddos_retirement"]

        same_event_report = self.engine.compare(irna_match.content, mehr_match.content)
        diff_event_report = self.engine.compare(irna_match.content, irna_ghoddos.content)

        self.assertGreater(
            same_event_report.hybrid_overall_score,
            diff_event_report.hybrid_overall_score + 0.25,
        )

    def test_custom_english_and_persian_news_comparison(self) -> None:
        custom_en_1 = "Arsenal defeated Chelsea 2-0 at Emirates Stadium thanks to two second-half goals from Bukayo Saka."
        custom_en_2 = "Bukayo Saka scored twice in the second half as Arsenal beat Chelsea two-nil at the Emirates Stadium."
        rep_en = self.engine.compare(custom_en_1, custom_en_2)
        self.assertGreater(rep_en.hybrid_overall_score, 0.50)


if __name__ == "__main__":
    unittest.main()
