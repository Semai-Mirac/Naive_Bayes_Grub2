Bu Python kodu, meme kanseri teşhisi için üç farklı Naive Bayes sınıflandırıcısını (Bernoulli, Gaussian ve Multinomial) uygulayan ve performanslarını değerlendiren bir makine öğrenimi projesidir. Kod, pandas, scikit-learn, matplotlib ve seaborn kütüphanelerini kullanır.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Kodun Adımları:

Veri Yükleme ve Ön İşleme:

"breast_cancer.csv" dosyasından veri yüklenir.
'Class' sütunundaki değerler, 2 (benign) ve 4 (malignant) olarak etiketlendikten sonra, 'benign' ve 'malignant' stringlerine dönüştürülür.
'Sample code number' ve 'Class' sütunları, özellikler (X) ve hedef değişken (y) olarak ayrılır.
Veri, eğitim ve test kümelerine bölünür.

Bernoulli Naive Bayes:

BernoulliNB sınıflandırıcısı oluşturulur ve binarize parametresi 5 olarak ayarlanır. Bu, özellik değerlerini 5'ten büyükse 1, değilse 0 olarak ikilileştirir.
Model, eğitim verileriyle eğitilir.
Test verileri üzerinde tahminler yapılır.
Doğruluk, kesinlik, geri çağırma ve F1 skoru gibi performans metrikleri hesaplanır ve yazdırılır.
Karışıklık matrisi oluşturulur ve ısı haritası olarak görselleştirilir.

Gaussian Naive Bayes:

GaussianNB sınıflandırıcısı oluşturulur.
Model, eğitim verileriyle eğitilir.
Test verileri üzerinde tahminler yapılır.
Performans metrikleri hesaplanır ve yazdırılır.
Karışıklık matrisi oluşturulur ve görselleştirilir.

Multinomial Naive Bayes:

MultinomialNB sınıflandırıcısı oluşturulur.
Model, eğitim verileriyle eğitilir.
Test verileri üzerinde tahminler yapılır.
Performans metrikleri hesaplanır ve yazdırılır.
Karışıklık matrisi oluşturulur ve görselleştirilir.

Grafikleri Gösterme:

Tüm karışıklık matrisi ısı haritaları ekrana gösterilir.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Kodun Özeti:

Bu kod, meme kanseri teşhisi için üç farklı Naive Bayes modelinin performansını karşılaştırmayı amaçlar. Her model için performans metrikleri hesaplanır ve karışıklık matrisleri görselleştirilir. Bu sayede, hangi modelin meme kanseri teşhisi için daha uygun olduğu değerlendirilebilir.  binarize parametresi ile BernoulliNB'nin verileri nasıl ikilileştirdiğine dikkat etmek önemlidir. MultinomialNB'nin metin sınıflandırmada sıklıkla kullanıldığı, bu veri setinin yapısı gereği GaussianNB ve BernoulliNB'nin daha uygun sonuçlar vermesi beklenebilir.  Kod, farklı Naive Bayes modellerinin güçlü ve zayıf yönlerini anlamak ve uygun modeli seçmek için bir temel sağlar.








