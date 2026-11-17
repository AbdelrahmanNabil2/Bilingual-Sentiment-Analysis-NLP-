# Data

Place dataset files in this folder. CSV files here are **ignored by git**.

For full schema, sizes, labels, and splits, see the root [`README.md`](../README.md#datasets).

## Files to add

### `arabic_tweets.csv`

Arabic tweet **sentiment** dataset (~18.7k usable rows).

| Required column | Values |
|-----------------|--------|
| `Tweet_text` | Arabic tweet text |
| `Sentiment_label` | `Negative` \| `Neutral` \| `Positive` |

Optional: tweet id, `Topic`, confidence scores, `Speech_act_label`, etc.

```bash
# if your export has a space in the name:
mv "arabic tweets.csv" arabic_tweets.csv
```

### `data.csv`

English **mental-health status** dataset (~52k rows).

| Required column | Values |
|-----------------|--------|
| `statement` | English text |
| `status` | Anxiety, Bipolar, Depression, Normal, Personality disorder, Stress, Suicidal |

## Colab

Upload both CSVs to `/content`, or copy them into this `data/` folder after cloning. The notebook auto-detects `../data` then `/content`.
