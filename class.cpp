#include <iostream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더

using namespace cv;
using namespace std;

Mat doDft(Mat srcImg);
Mat getMagnitude(Mat complexImg);
Mat myNormalize(Mat src);
Mat getPhase(Mat complexImg);
Mat centralize(Mat complex);
Mat setComplex(Mat magImg, Mat phaImg);
Mat doIdft(Mat complexImg);
Mat padding(Mat img);
Mat doLPF(Mat srcImg);
Mat doHPF(Mat srcImg);

void ex1();
void ex2();

int main() {
	ex2();

	return 0;
}

void ex1() {
	Mat src_img = imread("img1.jpg", 0);
	Mat dst_img = doDft(src_img);
	dst_img = centralize(dst_img);
	dst_img = getMagnitude(dst_img);
	dst_img = myNormalize(dst_img);

	imshow("src_img", src_img);
	imshow("dst_img", dst_img);
	waitKey(0);
	destroyAllWindows();
}

void ex2() {
	Mat src_img = imread("img1.jpg", 0);
	Mat dst_img = doDft(src_img);
	dst_img = centralize(dst_img);
	dst_img = getPhase(dst_img);
	dst_img = myNormalize(dst_img);

	imshow("src_img", src_img);
	imshow("dst_img", dst_img);
	waitKey(0);
	destroyAllWindows();
}

Mat doDft(Mat srcImg) {
	Mat floatImg;
	srcImg.convertTo(floatImg, CV_32F);

	Mat complexImg;
	dft(floatImg, complexImg, DFT_COMPLEX_OUTPUT);

	return complexImg;
}

Mat getMagnitude(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes);

	Mat magImg;
	magnitude(planes[0], planes[1], magImg);
	magImg += Scalar::all(1);
	log(magImg, magImg);

	return magImg;
}

Mat myNormalize(Mat src) {
	Mat dst;
	src.copyTo(dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1);

	return dst;
}

Mat getPhase(Mat complexImg){
	Mat planes[2];
	split(complexImg, planes);

	Mat phaImg;
	phase(planes[0], planes[1], phaImg);

	return phaImg;
}

Mat centralize(Mat complex){
	Mat planes[2];
	split(complex, planes);
	int cx = planes[0].cols / 2;
	int cy = planes[1].rows / 2;

	Mat q0Re(planes[0], Rect(0, 0, cx, cy));
	Mat q1Re(planes[0], Rect(cx, 0, cx, cy));
	Mat q2Re(planes[0], Rect(0, cy, cx, cy));
	Mat q3Re(planes[0], Rect(cx, cy, cx, cy));

	Mat tmp;
	q0Re.copyTo(tmp);
	q3Re.copyTo(q0Re);
	tmp.copyTo(q3Re);
	q1Re.copyTo(tmp);
	q2Re.copyTo(q1Re);
	tmp.copyTo(q2Re);

	Mat q0Im(planes[1], Rect(0, 0, cx, cy));
	Mat q1Im(planes[1], Rect(cx, 0, cx, cy));
	Mat q2Im(planes[1], Rect(0, cy, cx, cy));
	Mat q3Im(planes[1], Rect(cx, cy, cx, cy));

	q0Im.copyTo(tmp);
	q3Im.copyTo(q0Im);
	tmp.copyTo(q3Im);
	q1Im.copyTo(tmp);
	q2Im.copyTo(q1Im);
	tmp.copyTo(q2Im);

	Mat centerComplex;
	merge(planes, 2, centerComplex);

	return centerComplex;
}

Mat setComplex(Mat magImg, Mat phaImg){
	exp(magImg, magImg);
	magImg -= Scalar::all(1);

	Mat planes[2];
	polarToCart(magImg, phaImg, planes[0], planes[1]);

	Mat complexImg;
	merge(planes, 2, complexImg);

	return complexImg;
}

Mat doIdft(Mat complexImg){
	Mat idftcvt;
	idft(complexImg, idftcvt);

	Mat planes[2];
	split(idftcvt, planes);

	Mat dstImg;
	magnitude(planes[0], planes[1], dstImg);
	normalize(dstImg, dstImg, 255, 0, NORM_MINMAX);
	dstImg.convertTo(dstImg, CV_8UC1);

	return dstImg;
}

Mat padding(Mat img) {
	int dftRows = getOptimalDFTSize(img.rows);
	int dftCols = getOptimalDFTSize(img.cols);

	Mat padded;
	copyMakeBorder(img, padded, 0, dftRows - img.rows, 0, dftCols - img.cols, BORDER_CONSTANT, Scalar::all(0));
	return padded;
}


Mat doLPF(Mat srcImg){
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 20, Scalar::all(1), -1, -1, 0);

	Mat magImg2;
	multiply(magImg, maskImg, magImg2);

	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);

}

Mat doHPF(Mat srcImg) {
	// <DFT>
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	// <LFT>
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal,&maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg = Mat::ones(magImg.size(), CV_32F);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 50, Scalar::all(0), -1, -1, 0);

	Mat magImg2;
	multiply(magImg, maskImg, magImg2);

	// <IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);

}