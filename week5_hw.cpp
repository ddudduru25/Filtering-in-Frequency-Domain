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

Mat doBPF(Mat src_img);
Mat spatialSF(Mat srcImg);
int myKernelConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height);
Mat rmFlicker(Mat src_img);

void ex1();
void ex2();

int main() {

	/*
	//img1.jpg에 band pass filter를 적용
	Mat src_img = imread("img1.jpg", 0);
	imshow("src_img", src_img);

	Mat dst_img;
	dst_img = doBPF(src_img);
	imshow("dst_img", dst_img);

	waitKey(0);
	destroyAllWindows();
	*/

	/*
	//Spatial domain, frequency domain 각각에서 sobel filter를 구현하고 img2.jpg에 대해 비교
	Mat src_img = imread("img2.jpg", 0);
	imshow("src_img", src_img);

	Mat dst_img1;
	dst_img1 = spatialSF(src_img);
	imshow("SpatialDomain", dst_img1);

	Mat dst_img2;
	dst_img2 = doHPF(src_img);
	imshow("FrequencyDomain", dst_img2);

	waitKey(0);
	destroyAllWindows();
	*/

	//img3.jpg에서 나타나는 flickering 현상을 frequency domain filtering을 통해 제거
	Mat src_img = imread("img3.jpg", 0);
	imshow("src_img", src_img);

	Mat dst_img;
	dst_img = rmFlicker(src_img);
	imshow("dst_img", dst_img);
	waitKey(0);
	destroyAllWindows();
	

	return 0;
}

Mat doBPF(Mat src_img) {
	Mat padImg = padding(src_img);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX); 

	Mat maskImg = Mat::zeros(magImg.size(), CV_32F); 
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 100, Scalar::all(1), -1, -1, 0); 
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 20, Scalar::all(0), -1, -1, 0); 

	Mat maskImg2;
	multiply(magImg, maskImg, maskImg2); 
	imshow("mag_img2", maskImg2); 

	normalize(maskImg2, maskImg2, (float)minVal, (float)maxVal, NORM_MINMAX); 

	Mat complexImg2 = setComplex(maskImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);
}

Mat spatialSF(Mat srcImg) { //3주차 실습 mySobelFilter 참고
	int kernelX[3][3] = { -1, 0, 1,
						-2, 0, 2,
						-1, 0, 1 };
	int kernelY[3][3] = { -1, -2, -1,
						0, 0, 0,
						1, 2, 1 };
	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;
	int width = srcImg.cols;
	int height = srcImg.rows;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y * width + x] = (abs(myKernelConv3x3(srcData, kernelX, x, y, width, height)) +
				abs(myKernelConv3x3(srcData, kernelY, x, y, width, height))) / 2;
		}
	}
	return dstImg;
}

int myKernelConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height) {
	//3주차 실습 myKernelConv3x3함수 참고
	
	int sum = 0;
	int sumKernel = 0;

	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + 1) < width) {
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}

	if (sumKernel != 0) return sum / sumKernel;
	else return sum;
}

Mat rmFlicker(Mat src_img) {
	Mat padImg = padding(src_img);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg = Mat::ones(magImg.size(), CV_32F);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 30, Scalar::all(0), -1, -1, 0);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 5, Scalar::all(1), -1, -1, 0); 

	Mat magImg2;
	multiply(magImg, maskImg, magImg2);
	imshow("mask", magImg2);

	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);
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

Mat getPhase(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes);

	Mat phaImg;
	phase(planes[0], planes[1], phaImg);

	return phaImg;
}

Mat centralize(Mat complex) {
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

Mat setComplex(Mat magImg, Mat phaImg) {
	exp(magImg, magImg);
	magImg -= Scalar::all(1);

	Mat planes[2];
	polarToCart(magImg, phaImg, planes[0], planes[1]);

	Mat complexImg;
	merge(planes, 2, complexImg);

	return complexImg;
}

Mat doIdft(Mat complexImg) {
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


Mat doLPF(Mat srcImg) {
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
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
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