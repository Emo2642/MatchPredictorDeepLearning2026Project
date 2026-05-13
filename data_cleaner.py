import pandas as pd

# Dosyaları oku
matches = pd.read_csv("Matches_Top5_TR.csv", low_memory=False)
elo = pd.read_csv("EloRatings_Top5_TR.csv")

# Tarih kolonlarını datetime yap
matches["MatchDate"] = pd.to_datetime(matches["MatchDate"], errors="coerce")
elo["date"] = pd.to_datetime(elo["date"], errors="coerce")

# 2015-2025 arası filtre
start_date = "2015-01-01"
end_date = "2025-12-31"

matches_clean = matches[
    (matches["MatchDate"] >= start_date) &
    (matches["MatchDate"] <= end_date)
].copy()

elo_clean = elo[
    (elo["date"] >= start_date) &
    (elo["date"] <= end_date)
].copy()

# Temizlenmiş dosyaları kaydet
matches_clean.to_csv("Matches_Top5_TR_2015_2025.csv", index=False)
elo_clean.to_csv("EloRatings_Top5_TR_2015_2025.csv", index=False)

print("Matches eski boyut:", matches.shape)
print("Matches yeni boyut:", matches_clean.shape)

print("Elo eski boyut:", elo.shape)
print("Elo yeni boyut:", elo_clean.shape)

print("Temizleme tamamlandı.")