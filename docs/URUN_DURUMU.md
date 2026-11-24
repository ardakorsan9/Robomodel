# Ürün durumu ve eksikler (2026-11-24)

Bu liste testlere, tarayıcı denemesine ve gerçek robotlarla yapılan karşılaştırmalara dayanıyor.
Kullandıkça bulunan eksikler buraya eklenecek.

## Hazır olanlar

| Alan | Durum |
|---|---|
| Robot tipleri | Sualtı (AUV, ROV), su üstü (tekne, USV), drone (çok rotorlu), kara robotu (tekerlekli, paletli), robot kol (web + komut satırı); bacaklı robotlar sadece komut satırında |
| CAD girişi | STL, STEP (montaj parça parça), OBJ, PLY; birim, eksen, döndürme; bozuk CAD onarımı ve voksel yedeği |
| Kütle dağılımı | STEP montajında parça adından malzeme tahmini, toplam kütleye ölçekleme, makul değilse uyarı; tek parçada eşit yoğunluk |
| Çıktılar | MATLAB (alt/orta/üst model), Simulink kurucu, Gazebo (yalnız sualtı), URDF (kol), toolbox gerektirmeyen MATLAB kol dinamiği, rapor, YAML |
| Web | Sade akış (tip, dosya, kütle), isteğe bağlı ayrıntılar, gereksinim kartları (gereken itki/tork), motor listesi (T200, Dynamixel), tıklayarak yerleştirme, TR/EN, 2 saatte otomatik silme |
| Testler | 253 otomatik test; MATLAB ile Python farkı 10⁻⁷–10⁻¹⁵ (sualtı, su üstü, drone, kara, kol) |

## Ölçülen doğruluk

| Değer | Hata | Kaynak |
|---|---|---|
| Hacim, kaldırma, GM, su çekimi | %0–4 | REMUS 100, kutu duba, Wigley |
| Tekerlekli robot son hız / dönüş | %4–11 | TurtleBot3 (ROBOTIS) |
| Kol yerçekimi torku (ana eklemler) | %14–21 (üreticinin kendi değerleri %10–16) | Franka Panda (Gaz et al. 2019) |
| Atalet, eşit yoğunlukla | %0–45 | TurtleBot3, Panda uç eklemler |
| Su / hava direnci (sönüm) | 1,6–10 kat düşük | REMUS 100, BlueROV2 |
| Gemi dönüş katsayıları (N_r, Y_r) | −%45 / eksik | Wigley, Clarke 1983 |

## Eksikler (öncelik sırasıyla)

### A. Doğruluk (yeni özellikten önce)
1. **Sönüm düzeltmesi:** yayınlanmış ölçümlerden şekil türüne göre katsayı. Yarım kalan çalışma ayrı
   bir dalda; bitirilip birleştirilecek.
2. **Kalibrasyonu tüm tiplere ve web'e taşımak:** şu an yalnızca sualtı için komut satırında
   (`robomodel calibrate`).
3. **Rotor ve dişli ataleti (kol eklemleri), motor/dişli malzeme tahmini:** Panda'nın uç eklem hatasının
   kaynağı.
4. **Doğruluk eşiklerini testlere yazmak:** REMUS, BlueROV2, TurtleBot3, Panda.
5. **Bekleyen doğrulamalar:** Crazyflie (drone; ETH tezi ve CAD gerekli), KCS gemisi (SIMMAN verisi).

### B. Kullanım
6. **Yeni sade sayfayı tarayıcıda denemek** (gereksinim kartları, motor listesi, kapalı bölümler).
7. **Motor verisi girilmemiş tekerlek** için MATLAB çıktısını MATLAB'da denemek.
8. **STEP montajından itici, rotor ve tekerleği otomatik bulmak** (parça adlarından).
9. **Drone motor+pervane itki tablosu** (kaynaklı değerler); KV + pervane boyundan tahmin.
10. **Komut satırı raporuna gereksinim bölümü** (şu an yalnızca web raporunda).
11. **KILAVUZ.md'yi yeni sayfaya göre güncellemek.**

### C. Eksik fizik ve çıktılar
12. **Gazebo çıktısı:** su üstü, drone, kara robotu.
13. **Drone:** itkinin hava hızıyla düşmesi, motor gecikmesi, jiroskopik etki (son hız ve tırmanma şu an
    üst sınır).
14. **Kara robotu:** yük aktarımı, eğim, süspansiyon.
15. **Su üstü:** dalga, Froude 0,3 üstünde direnç.
16. **Kara robotu ve kol için doğrusal model** (kontrolcü tasarımı için A, B).

### D. Yayın
17. **Sunucu, alan adı, HTTPS:** Docker hazır ama denenmedi.
18. **Gizlilik metni ve kullanım koşulları:** kullanıcı CAD'leri için.
19. **Windows ve macOS'ta kurulum denemesi.**
20. **Kulüpten kullanıcı testi:** 3–5 kişi, kendi CAD'leriyle.
