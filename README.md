# SASRec Spotify Playlist Predictor

This project implements a transformer-based sequential recommendation system (SASRec) trained on the Spotify Million Playlist Dataset (MPD). It predicts the next track in a playlist based on the previous listening history using masked self-attention.

## Features

* Converts raw Spotify MPD JSON into user–track interaction sequences
* Implements SASRec (Self-Attentive Sequential Recommendation)
* Trains on historical sequences to model user behavior
* Outputs top-10 next-track predictions based on playlist context

---

## Dataset

* **Source**: Spotify MPD (Recsys Challenge 2018)
* **Format**: `mpd.slice.0-999.json`
* The script processes this JSON into a CSV:

  ```csv
  user_id,track_id,timestamp
  ```

---

## Model: SASRec

* Transformer encoder with masked self-attention
* Positional embeddings capture sequence order
* Trained with cross-entropy loss for next-item prediction
* Designed for fast inference and personalization

### Architecture

* Embedding layer for items and positions
* 2-layer TransformerEncoder with 2 attention heads
* Output projection to item vocabulary

---

## Sample Output

```
Top-10 recommendations: [38182, 12437, 51999, 8021, 1203, 987, 4422, 10201, 7741, 3981]
```

---

## Evaluation 

![image](https://github.com/user-attachments/assets/fb2beeab-084e-4087-acc0-76bdfbe49feb)

- **Recall@10**: 0.42  
  > Measures how often the correct next track appears in the top 10 predictions. A score of 0.42 means the model got it right 42% of the time.

- **NDCG@10 (Normalized Discounted Cumulative Gain)**: 0.36  
  > Evaluates not just if the correct track was predicted, but how high it ranked. A higher NDCG means more relevant tracks are ranked closer to the top.

- **Inference Latency**: ~22 ms per user sequence  
  > Time taken by the model to generate recommendations for a single playlist sequence—optimized for real-time performance.

---

## Reference

Based on the [SASRec paper (2018)](https://arxiv.org/abs/1808.09781) by Amazon Research. This implementation uses a simplified encoder-only Transformer architecture tailored for next-item prediction tasks on sequential music data.
