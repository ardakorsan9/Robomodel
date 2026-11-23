# Doğrulama: kara robotu (tekerlekli, paletli)

Testler: `tests/test_ground.py` · Örnek: `examples/rover.yaml` · Kod: `src/robomodel/ground.py`

## Model

Düz zeminde düzlemsel hareket. Durum vektörü `x = [x, y, ψ, u, v, r]`: konum, yön ve ağırlık
merkezinin (CG) gövde hızları.

```
m (u̇ − v r) = Σ Fx      m (v̇ + u r) = Σ Fy      I_z ṙ = Σ (x_i Fy_i − y_i Fx_i)
```

**CAD ve parça listesinden çıkanlar:**
- kütle,
- CG ve onun **yerden yüksekliği**,
- dönme ataleti I_z,
- tekerlek yerleşimi,
- her tekerleğe düşen statik yük. Üçten fazla tekerlekte yük dağılımı statik olarak
  belirsizdir; rijit şasi için en düzgün dağılım alınıyor.

**Her temas noktasındaki kuvvet:**
- **Tahrik:** DC motorun düz tork-hız doğrusu, τ = τ_durma (komut − ω/ω₀). Tekerleğin kaymadan
  döndüğü varsayılıyor. Motor değerleri tekerlekte, yani dişli kutusundan sonra veriliyor.
- **Yuvarlanma direnci:** C_rr·N.
- **Yan kuvvet:** sürtünme μN. Kayma açısıyla büyüyor (~6°'de tepeye yakın), durma anına
  yakın yumuşatılmış Coulomb sürtünmesi kullanılıyor. Sarhoş tekerlekte yan kuvvet yok.
- **Sürtünme çemberi:** toplam kuvvet μN ile sınırlı.

**Palet:** 5 temas noktasına bölünüyor; motor kuvveti bu noktalara eşit paylaştırılıyor.

## Analitik doğrulamalar

| Durum | Beklenen (el hesabı) | Sonuç |
|---|---|---|
| Simetrik 4 tekerlek | her tekerlek W/4 | tam |
| 3 tekerlek (2 tahrikli + sarhoş) | kaldıraç kuralı: N_sarhoş = W·x_g / dingil mesafesi | tam |
| Devrilme açısı | atan(d / h), d: CG'nin kenara uzaklığı, h: CG yüksekliği | tam |
| Son hız | Σ F_s (1 − v/v₀) = C_rr W | 10⁻⁹ (formül), 10⁻⁴ (simülasyon) |
| Kalkış, güçlü motor | sürtünme sınırlı: a = μ g | 10⁻⁹ |
| Diferansiyel sürüş, yavaş | r = (v_sol − v_sağ) / iz genişliği | %1 |
| Ön tekerlekten yönlendirmeli, arkadan tahrikli | bisiklet modeli r = u tan δ / L | %5 |
| Palet | 10 temas noktası, yükler toplamı W | tam |
| Serbest yuvarlanma | yuvarlanma direnci robotu durdurur, enerji hep azalır | doğru |

Testlerin gösterdiği **fiziksel etki:** hız artınca dönüş, kinematiğin söylediğinden geniş
olur. CG aksın önünde ve sarhoş teker yan kuvvet vermiyorsa, merkezcil kuvveti yalnızca aks
sağlar. Bu bir yaw momenti doğurur ve motorların yük altında yavaşlaması dönüşü açar. Gerçek
robotlarda da görülen bir durum; model bunu kendiliğinden veriyor.

MATLAB (ode15s) ile Python (LSODA) arasındaki fark, 5 s'lik eğri bir sürüşte 1,3·10⁻⁷.

## Neye güvenilir?

- **Güvenilir:** tekerlek yükleri, devrilme açıları ve ivme sınırları, kinematik (dönüş
  yarıçapı, diferansiyel dönüş), motor verisi doğruysa son hız. Hepsi CG'nin konumuna ve
  yüksekliğine bağlı: bataryayı ve motorları parça olarak listeleyin.
- **Tahmin:** μ ve C_rr, zemin türüne göre tipik değerler (beton 0,8 / 0,015, çim 0,45 / 0,1
  gibi). Low/high modelleri 0,7–1,2·μ, 0,5–2·C_rr ve ±%15 motor aralığını kapsıyor. Yaylı
  kantarla yapılacak bir çekme testi ikisini de ölçer.
- **Paletli ve kayarak dönen (skid-steer) robotların yerinde dönüşü:** sürtünme modeline çok
  bağlı. Değeri ölçüp karşılaştırın.
- **Modelde yok:** ivmelenirken ve dönerken yük aktarımı, tekerlek dönme ataleti,
  süspansiyon, eğim ve tümsekler, motor elektrik dinamiği. Gazebo/URDF çıktısı kara robotu
  için henüz bağlanmadı (`robomodel robot` ile eklemli model ayrı olarak kurulabilir).

## Gerçek veriyle karşılaştırma: yapılmadı

Bunun için bir kara robotunda düz sürüşte hız, yerinde dönüş hızı ve çekme kuvveti ölçmek
yeterli. Kulübün gezgin robotu (rover) bunun için uygun bir aday.
