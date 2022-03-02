* 파이썬 설치: https://dlehdgml0480.tistory.com/8
* opencv-contrib-python 설치 필요
* ubuntu imagemagick 설치시 policy.xml에 pdf권한 부여 필요
  * 위치: /etc/ImageMagick-6/policy.xml 
    * \<policy domain="coder" rights="read | write" pattern="PDF" />
    * PDF: none -> read|write