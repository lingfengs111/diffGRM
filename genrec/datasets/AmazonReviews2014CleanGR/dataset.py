"""CleanGR-prepared Amazon Reviews 2014 adapter.

This reuses the prepared-data reader used for Amazon 2023 so the Beauty-2014
DiffGRM runs consume exactly the same item vocabulary and JSONL examples as the
CleanGR SASRec/TIGER baselines.
"""

from genrec.datasets.AmazonReviews2023CleanGR.dataset import AmazonReviews2023CleanGR


class AmazonReviews2014CleanGR(AmazonReviews2023CleanGR):
    """Load a CleanGR-prepared Amazon Reviews 2014 sequential dataset."""

    amazon_release = "2014"
