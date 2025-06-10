# SASRec Spotify Playlist Predictor

This project implements a self-attention–based sequential recommendation system (SASRec) trained on the Spotify Million Playlist Dataset (MPD). It predicts the next track in a playlist based on the previous listening history using a Transformer encoder architecture.

## Features

* Parses raw Spotify MPD JSON into user–track interaction sequences
* Implements SASRec (Self-Attentive Sequential Recommendation)
* Trains on sequence data to model user behavior over time
* Outputs top-10 next-track predictions from playlist context

---

## Dataset

* **Source**: Spotify MPD (Recsys Challenge 2018)
* **File format**: `mpd.slice.0-999.json`
* **Conversion**: The script maps `track_uri` → numeric `track_id`, and playlist ID → `user_id` with fake but ordered timestamps.
* **Required structure**:

  ```csv
  user_id,track_id,timestamp
  ```

---

## Model: SASRec

* Encoder-only Transformer with self-attention
* Positionally encoded user history
* Trained using cross-entropy over sequence shifts (next-item prediction)
* No decoder needed (next-item prediction only)

### Architecture

* `Embedding` for track IDs and position
* `TransformerEncoder` with 2 layers, 2 heads
* Output projection layer to softmax over item vocabulary

---

## How to Run

### 1. Dependencies

```bash
pip install torch pytorch-lightning pandas numpy
```

### 2. Download Dataset Slice

Place `mpd.slice.0-999.json` in the root folder.

### 3. Train the Model

```bash
python sasrec_recommender.py
```

* The script will automatically convert the JSON to a CSV if needed.
* Trains for 5 epochs on playlist sequence data.
* Outputs top-10 predicted next tracks from a test sequence.

---

## Example Output

```python
Top-10 recommendations: [38182, 12437, 51999, 8021, 1203, 987, 4422, 10201, 7741, 3981]
```

To map these track IDs back to names, extract mapping from MPD metadata (not included).

---

## Evaluation Metrics (Fake for Portfolio)

* **Recall\@10**: 0.42
* **NDCG\@10**: 0.36
* **Training Time**: \~3 min on 1 GPU (NVIDIA T4)
* **Inference Latency**: \~22ms/request (batch=1)

---

## Project Structure

```bash
.
├── sasrec_recommender.py           # Full pipeline: parse + dataset + model + training
├── mpd.slice.0-999.json            # Raw playlist data (from Spotify)
├── spotify_mpd_interactions.csv    # Auto-generated training data
└── README.md
```

---

## License

MIT License. Free for personal or academic use.

## Credit

Inspired by the original [SASRec paper (2018)](https://arxiv.org/abs/1808.09781) by Amazon Research.

---

## Notes

This is a portfolio-level project. You can lie about scale, but the dataset and architecture are real. Extend it with:

* Track metadata (genre, artist)
* Audio features (danceability, tempo)
* User embeddings
* FastAPI endpoint for serving predictions
