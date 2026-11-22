# Eksen ve işaret kuralları

Bu dosya projenin en önemli kuralıdır. `auv_gazebo` projesindeki sorunların çoğu
(işaret hataları, Gazebo ve MATLAB eksenlerinin karışması) bu kuralların tek bir yerde
tanımlı olmamasından çıktı.

## Kural: içeride tek standart, dönüşüm sadece dışa aktarımda

Yazılımın içindeki her hesap **Fossen / SNAME** standardını kullanır. Başka bir eksen
sistemine (Gazebo, URDF) dönüşüm sadece `export` modüllerinde, tek bir fonksiyonla yapılır.

## Gövde ekseni (body frame)

| Eksen | Yön | Hız | Açısal hız | Kuvvet | Moment |
|---|---|---|---|---|---|
| x | ileri (burun) | u (surge) | p (roll) | X | K |
| y | sağ (sancak) | v (sway) | q (pitch) | Y | M |
| z | aşağı | w (heave) | r (yaw) | Z | N |

- Orijin: geometri dosyasının orijini. Primitive şekillerde (küre, elipsoit, silindir, Myring)
  gövdeyi saran kutunun merkezi; uzun eksen x (burun +x yönünde).
- Birimler: metre, kilogram, saniye, radyan. YAML'da `unit: mm` sadece mesh dosyası için.
- `parts` ve `thrusters` içindeki tüm konumlar bu eksende verilir.
- Örnek: aracın arkasındaki itici, ileri iterken: `position: [-0.8, 0, 0]`, `direction: [1, 0, 0]`.

## Dünya ekseni (NED)

| Eksen | Yön |
|---|---|
| x | kuzey |
| y | doğu |
| z | aşağı (derinlik pozitif) |

Euler açıları: φ (roll), θ (pitch), ψ (yaw), sıra z-y-x (önce yaw).

## Hidrodinamik katsayıların işareti

- Fossen notasyonu: eklenmiş kütle `X_u̇`, sönümleme `X_u`, `X_u|u|` vb.
- **Eklenmiş kütle türevleri negatiftir** (`X_u̇ < 0`). Eklenmiş kütle matrisi
  `M_A = -diag(X_u̇, Y_v̇, Z_ẇ, K_ṗ, M_q̇, N_ṙ)` pozitif tanımlıdır.
- **Sönümleme katsayıları negatiftir** (`X_u < 0`, `X_u|u| < 0`).
  Sönümleme matrisi `D(ν)` pozitif tanımlıdır.
- Kaynaktan (makale, tez) değer alınırken işaret kuralı mutlaka kontrol edilir ve
  kaynağın kuralı yorum satırında yazılır.

## Dışa aktarım dönüşümleri

| Hedef | Eksen | Dönüşüm |
|---|---|---|
| MATLAB (MSS Toolbox) | Fossen ile aynı | Yok |
| Gazebo / URDF / ROS | x ileri, y sol, z yukarı (FLU), dünya ENU | x aynı, y ve z işaret değiştirir; çapraz terimler buna göre |

Dönüşüm fonksiyonu yazıldığında testleri burada listelenecek.
