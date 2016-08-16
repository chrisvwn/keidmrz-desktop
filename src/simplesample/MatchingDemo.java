package simplesample;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.Imgproc;

import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.RenderedImage;
import java.io.IOException;
import java.math.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import cern.colt.*;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Matrix2DMatrix2DFunction;

class MatchingDemo {
    // The reference image (this detector's target).
    private final Mat mReferenceImage = Imgcodecs.imread("/home/chris/Pictures/Webcam/id-back2.jpg");

    // Features of the reference image.
    private final MatOfKeyPoint mReferenceKeypoints = new MatOfKeyPoint();

    // Descriptors of the reference image's features.
    private final Mat mReferenceDescriptors = new Mat();

    // The corner coordinates of the reference image, in pixels.
    // CvType defines the color depth, number of channels, and
    // channel layout in the image. Here, each point is represented
    // by two 32-bit floats.
    private final Mat mReferenceCorners = new Mat(4, 1, CvType.CV_32FC2);

    // Features of the scene (the current frame).
    private final MatOfKeyPoint mSceneKeypoints = new MatOfKeyPoint();

    // Descriptors of the scene's features.
    private final Mat mSceneDescriptors = new Mat();

    // Tentative corner coordinates detected in the scene, in pixels.
    private final Mat mCandidateSceneCorners = new Mat(4, 1, CvType.CV_32FC2);

    // Good corner coordinates detected in the scene, in pixels.
    private final Mat mSceneCorners = new Mat(0, 0, CvType.CV_32FC2);

    // The good detected corner coordinates, in pixels, as integers.
    private final MatOfPoint mIntSceneCorners = new MatOfPoint();

    // A grayscale version of the scene.
    private final Mat mGraySrc = new Mat();

    // Tentative matches of scene features and reference features.
    private final MatOfDMatch mMatches = new MatOfDMatch();

    // A feature detector, which finds features in images.
    private final FeatureDetector mFeatureDetector = FeatureDetector.create(FeatureDetector.ORB);

    // A descriptor extractor, which creates descriptors of features.
    private final DescriptorExtractor mDescriptorExtractor =
            DescriptorExtractor.create(DescriptorExtractor.ORB);

    // A descriptor matcher, which matches features based on their descriptors.
    private final DescriptorMatcher mDescriptorMatcher =
            DescriptorMatcher.create(
                    DescriptorMatcher.BRUTEFORCE_HAMMINGLUT);

    // The color of the outline drawn around the detected image.
    private final Scalar mLineColor = new Scalar(0, 255, 0);

	
	public Mat resize(Mat img, int width, int height, int inter){
	
		inter = Imgproc.INTER_AREA;
		
		Size imgDim = img.size();
		
		Size dim = null;
		
		double r = 1;
		
		if(width <= 0 && height <= 0)
			return img;
		
		if (height == 0)
		{
			r =  width/imgDim.width;
			dim = new Size(width, (int)(img.height() * r));
		}
		else if(width == 0)
		{
			r = height/imgDim.height;
			dim = new Size((int)(img.width() * r), height);	
		}
		else if (width > 0 && height > 0)
		{
			dim = new Size(width, height);
		}
		
		
		
		//resize the image
	    Mat resized = new Mat();
	    
	    Imgproc.resize(img, resized, dim, 0, 0, inter);

		
		return resized;
	}
	
	public void displayImage(Image img2, String label)
	{   
		//BufferedImage img=ImageIO.read(new File("/HelloOpenCV/lena.png"));
		ImageIcon icon=new ImageIcon(img2);
		
		JFrame frame=new JFrame(label);
		
		frame.setLayout(new FlowLayout());        
		
		frame.setSize(img2.getWidth(null)+50, img2.getHeight(null)+50);     
		
		JLabel lbl=new JLabel();
		
		lbl.setIcon(icon);
		
		frame.add(lbl);
		
		frame.setVisible(true);
		
		frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
	}
	   
	public Image toBufferedImage(Mat m){
	    int type = BufferedImage.TYPE_BYTE_GRAY;
	    if ( m.channels() > 1 ) {
	        Mat m2 = new Mat();
	        Imgproc.cvtColor(m,m2,Imgproc.COLOR_BGR2RGB);
	        type = BufferedImage.TYPE_3BYTE_BGR;
	        m = m2;
	    }
	    byte [] b = new byte[m.channels()*m.cols()*m.rows()];
	    m.get(0,0,b); // get all the pixels
	    BufferedImage image = new BufferedImage(m.cols(),m.rows(), type);
	    image.getRaster().setDataElements(0, 0, m.cols(),m.rows(), b);
	    return image;

	}
	
	public MatchingDemo() throws IOException {

		// Create grayscale and RGBA versions of the reference image.
		final Mat referenceImageGray = new Mat();
		Imgproc.cvtColor(mReferenceImage, referenceImageGray,
		Imgproc.COLOR_BGR2GRAY);
		Imgproc.cvtColor(mReferenceImage, mReferenceImage,
		Imgproc.COLOR_BGR2RGBA);
		
		// Store the reference image's corner coordinates, in pixels.
		mReferenceCorners.put(0, 0, new double[] {0.0, 0.0});
		mReferenceCorners.put(1, 0, new double[] {referenceImageGray.cols(), 0.0});
		mReferenceCorners.put(2, 0, new double[] {referenceImageGray.cols(),referenceImageGray.rows()});
		mReferenceCorners.put(3, 0, new double[] {0.0, referenceImageGray.rows()});
		
		// Detect the reference features and compute their
		// descriptors.
		mFeatureDetector.detect(referenceImageGray, mReferenceKeypoints);
		mDescriptorExtractor.compute(referenceImageGray, mReferenceKeypoints, mReferenceDescriptors);
	}
	
	private void findSceneCorners() {

        final List<DMatch> matchesList = mMatches.toList();
        if (matchesList.size() < 4) {
            // There are too few matches to find the homography.
            return;
        }

        final List<KeyPoint> referenceKeypointsList =
                mReferenceKeypoints.toList();
        final List<KeyPoint> sceneKeypointsList =
                mSceneKeypoints.toList();

        // Calculate the max and min distances between keypoints.
        double maxDist = 0.0;
        double minDist = Double.MAX_VALUE;
        for (final DMatch match : matchesList) {
            final double dist = match.distance;
            if (dist < minDist) {
                minDist = dist;
            }
            if (dist > maxDist) {
                maxDist = dist;
            }
        }

        // The thresholds for minDist are chosen subjectively
        // based on testing. The unit is not related to pixel
        // distances; it is related to the number of failed tests
        // for similarity between the matched descriptors.
        if (minDist > 50.0) {
            // The target is completely lost.
            // Discard any previously found corners.
            mSceneCorners.create(0, 0, mSceneCorners.type());
            return;
        } else if (minDist > 25.0) {
            // The target is lost but maybe it is still close.
            // Keep any previously found corners.
            return;
        }

        // Identify "good" keypoints based on match distance.
        final ArrayList<Point> goodReferencePointsList =
                new ArrayList<Point>();
        final ArrayList<Point> goodScenePointsList =
                new ArrayList<Point>();
        final double maxGoodMatchDist = 1.75 * minDist;
        for (final DMatch match : matchesList) {
            if (match.distance < maxGoodMatchDist) {
                goodReferencePointsList.add(
                        referenceKeypointsList.get(match.trainIdx).pt);
                goodScenePointsList.add(
                        sceneKeypointsList.get(match.queryIdx).pt);
            }
        }

        if (goodReferencePointsList.size() < 4 ||
                goodScenePointsList.size() < 4) {
            // There are too few good points to find the homography.
            return;
        }

        // There are enough good points to find the homography.
        // (Otherwise, the method would have already returned.)

        // Convert the matched points to MatOfPoint2f format, as
        // required by the Calib3d.findHomography function.
        final MatOfPoint2f goodReferencePoints = new MatOfPoint2f();
        goodReferencePoints.fromList(goodReferencePointsList);
        final MatOfPoint2f goodScenePoints = new MatOfPoint2f();
        goodScenePoints.fromList(goodScenePointsList);

        // Find the homography.
        final Mat homography = Calib3d.findHomography(
                goodReferencePoints, goodScenePoints);

        // Use the homography to project the reference corner
        // coordinates into scene coordinates.
        Core.perspectiveTransform(mReferenceCorners,
                mCandidateSceneCorners, homography);

        // Convert the scene corners to integer format, as required
        // by the Imgproc.isContourConvex function.
        mCandidateSceneCorners.convertTo(mIntSceneCorners,
                CvType.CV_32S);

        // Check whether the corners form a convex polygon. If not,
        // (that is, if the corners form a concave polygon), the
        // detection result is invalid because no real perspective can
        // make the corners of a rectangular image look like a concave
        // polygon!
        if (Imgproc.isContourConvex(mIntSceneCorners)) {
            // The corners form a convex polygon, so record them as
            // valid scene corners.
            mCandidateSceneCorners.copyTo(mSceneCorners);
        }
    }
	
	public Mat detectIDMRZ(Mat img)
	{
        Mat roi = null;

        Mat rectKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(13,5));

        Mat sqKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(21,21));
        
        if (img.width() > 800)
	        // load the image, resize it, and convert it to grayscale
	        img = resize(img, 800, 600, Imgproc.INTER_AREA);
        
        displayImage(toBufferedImage(img), "orig image resized");

        Mat gray = new Mat();
        
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);      
        
        displayImage(toBufferedImage(gray), "image in grayscale");
        
    	//smooth the image using a 3x3 Gaussian, then apply the blackhat
    	//morphological operator to find dark regions on a light background
    	Imgproc.GaussianBlur(gray, gray, new Size(3, 3), 0);
    	
    	displayImage(toBufferedImage(gray), "gaussian blur");
    	
    	Mat blackhat = new Mat();
    	Imgproc.morphologyEx(gray, blackhat, Imgproc.MORPH_BLACKHAT, rectKernel);
        
    	displayImage(toBufferedImage(blackhat), "blackhat");
    	
    	//compute the Scharr gradient of the blackhat image and scale the
    	//result into the range [0, 255]
    	Mat gradX = new Mat();
    	//gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    	Imgproc.Sobel(blackhat, gradX, CvType.CV_32F, 1, 0, -1, 1, 0);
    	//gradX = Matrix absolute(gradX)
    	
    	//displayImage(toBufferedImage(gradX), "sobel");
    	
    	//(minVal, maxVal) = (np.min(gradX), np.max(gradX))
    	MinMaxLocResult minMaxVal = Core.minMaxLoc(gradX);

		//gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    	gradX.convertTo(gradX,CvType.CV_8U,255.0/(minMaxVal.maxVal-minMaxVal.minVal),-255.0/minMaxVal.minVal);
    	
    	displayImage(toBufferedImage(gradX), "sobel converted to CV_8U");
    	
    	//apply a closing operation using the rectangular kernel to close
    	//gaps in between letters -- then apply Otsu's thresholding method
    	Imgproc.morphologyEx(gradX, gradX, Imgproc.MORPH_CLOSE, rectKernel);
    	
    	displayImage(toBufferedImage(gradX), "closing operation morphology");
    	
    	Mat thresh = new Mat();
    	Imgproc.threshold(gradX, thresh, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
    	
    	displayImage(toBufferedImage(thresh), "applied threshold");
    	
    	// perform another closing operation, this time using the square
    	// kernel to close gaps between lines of the MRZ, then perform a
    	// series of erosions to break apart connected components
    	Imgproc.morphologyEx(thresh, thresh, Imgproc.MORPH_CLOSE, sqKernel);
    	
    	displayImage(toBufferedImage(thresh), "another closing operation morphology");
    	
    	Imgproc.erode(thresh, thresh, new Mat(), new Point(-1,-1), 4);
    	
    	displayImage(toBufferedImage(thresh), "erode");
    	// during thresholding, it's possible that border pixels were
    	// included in the thresholding, so let's set 5% of the left and
    	// right borders to zero
    	int pRows = (int)(img.rows() * 0.05);
    	int pCols = (int)(img.cols() * 0.05);
    	
    	//thresh[:, 0:pCols] = 0;
    	//thresh.put(thresh.rows(), pCols, 0);
    	//thresh[:, image.cols() - pCols] = 0;
    	for (int i=0; i <= thresh.rows(); i++)
    		for (int j=0; j<=pCols; j++)
    			thresh.put(i, j, 0);
    	
    	//thresh[:, image.cols() - pCols] = 0;
    	for (int i=0; i <= thresh.rows(); i++)
    		for (int j=img.cols()-pCols; j<=img.cols(); j++)
    			thresh.put(i, j, 0);
    	
    	displayImage(toBufferedImage(thresh), "");
    	
    	// find contours in the thresholded image and sort them by their
    	// size
    	List<MatOfPoint> cnts = new ArrayList<MatOfPoint>(); 

    	Imgproc.findContours(thresh.clone(), cnts, new Mat(), Imgproc.RETR_EXTERNAL,
    		Imgproc.CHAIN_APPROX_SIMPLE);
    	
    	//cnts.sort(Imgproc.contourArea(contour));//, Imgproc.contourArea(cnts, true))
     
    	// loop over the contours
    	for (MatOfPoint c : cnts){
    		// compute the bounding box of the contour and use the contour to
    		// compute the aspect ratio and coverage ratio of the bounding box
    		// width to the width of the image
    		Rect bRect = Imgproc.boundingRect(c);
    		int x=bRect.x;
    		int y=bRect.y;
    		int w=bRect.width;
    		int h=bRect.height;
    		
    		int grWidth = gray.width();
    		
    		float ar = (float)w / (float)h;
    		float crWidth = (float)w / (float)grWidth;
     
    		// check to see if the aspect ratio and coverage width are within
    		// acceptable criteria
    		if (ar > 5 && crWidth > 0.75){
    			// pad the bounding box since we applied erosions and now need
    			// to re-grow it
    			int pX = (int)((x + w) * 0.03);
    			int pY = (int)((y + h) * 0.03);
    			x = x - pX;
    			y = y - pY;
    			w = w + (pX * 2); 
    			h = h + (pY * 2);
    			
    			// extract the ROI from the image and draw a bounding box
    			// surrounding the MRZ
    			
    			roi = new Mat(img, new Rect(x, y, w, h));
    			
    			Imgproc.rectangle(img, new Point(x, y), new Point(x + w, y + h), new Scalar(0, 255, 0), 2);
    			
    			
    			displayImage(toBufferedImage(img), "found mrz?");
    			
    			break;
    		}
    	}
    	
    	return roi;
	}
	
    protected void draw(final Mat src, final Mat dst) {

        if (dst != src) {
            src.copyTo(dst);
        }

        if (mSceneCorners.height() < 4) {
            // The target has not been found.

            // Draw a thumbnail of the target in the upper-left
            // corner so that the user knows what it is.

            // Compute the thumbnail's larger dimension as half the
            // video frame's smaller dimension.
            int height = mReferenceImage.height();
            int width = mReferenceImage.width();
            final int maxDimension = Math.min(dst.width(),
                    dst.height()) / 2;
            final double aspectRatio = width / (double)height;
            if (height > width) {
                height = maxDimension;
                width = (int)(height * aspectRatio);
            } else {
                width = maxDimension;
                height = (int)(width / aspectRatio);
            }

            // Select the region of interest (ROI) where the thumbnail
            // will be drawn.
            final Mat dstROI = dst.submat(0, height, 0, width);

            // Copy a resized reference image into the ROI.
            Imgproc.resize(mReferenceImage, dstROI, dstROI.size(),
                    0.0, 0.0, Imgproc.INTER_AREA);

            return;
        }

        // Outline the found target in green.
        Imgproc.line(dst, new Point(mSceneCorners.get(0, 0)),
                new Point(mSceneCorners.get(1, 0)), mLineColor, 4);
        Imgproc.line(dst, new Point(mSceneCorners.get(1, 0)),
                new Point(mSceneCorners.get(2, 0)), mLineColor, 4);
        Imgproc.line(dst, new Point(mSceneCorners.get(2, 0)),
                new Point(mSceneCorners.get(3, 0)), mLineColor, 4);
        Imgproc.line(dst, new Point(mSceneCorners.get(3,0)),
                new Point(mSceneCorners.get(0, 0)), mLineColor, 4);
    }
	
	static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }
	
    public void run(String inFile, String outFile) {
        System.out.println("\nRunning MRZ search ...");

        Mat img = Imgcodecs.imread(inFile);
        
//        URL ipwebcam = null;
//        
//		try {
//			ipwebcam = new URL("http://192.168.0.14:8080/photo.jpg");
//		} catch (MalformedURLException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//        
//        RenderedImage imgDnld = null;
//		try {
//			imgDnld = ImageIO.read(ipwebcam);
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//        
//        byte[] imgData = ((DataBufferByte) imgDnld.getData().getDataBuffer()).getData();
//        
//        if (imgData == null)
//        	return;
//        
//        Mat img = new Mat();
//        img.put(0, 0, imgData);
        
        displayImage(toBufferedImage(img), "orig image before scale");

        
        if (img.width() > 800)
	        while (img.width() > 800)// || img.height() > 600)
	        	Imgproc.pyrDown(img, img);
//        else
//        	while (img.width() < 800)
//        		Imgproc.pyrUp(img, img);
        
        displayImage(toBufferedImage(img), "orig image");
        
        //
        //START FIND REFERENCE IMAGE
        //
        
        // Convert the scene to grayscale.
        Imgproc.cvtColor(img, mGraySrc, Imgproc.COLOR_RGBA2GRAY);

        // Detect the scene features, compute their descriptors,
        // and match the scene descriptors to reference descriptors.
        mFeatureDetector.detect(mGraySrc, mSceneKeypoints);

        mDescriptorExtractor.compute(mGraySrc, mSceneKeypoints, mSceneDescriptors);
        
        mDescriptorMatcher.match(mSceneDescriptors, mReferenceDescriptors, mMatches);

        // Attempt to find the target image's corners in the scene.
        findSceneCorners();

        // If the corners have been found, draw an outline around the
        // target image.
        // Else, draw a thumbnail of the target image.
        Mat foundIDImg = new Mat();
        draw(img, foundIDImg);
        
		displayImage(toBufferedImage(foundIDImg), "found id?");

        //
        //END FIND REFERENCE IMAGE
        //
        
		Point topLeft = new Point(mSceneCorners.get(0, 0));
		
		Point bottomRight = new Point(mSceneCorners.get(2, 0));
		
		Point bottomLeft = new Point(mSceneCorners.get(1, 0));
		
		Point topRight = new Point(mSceneCorners.get(3, 0));
		
		//Mat detectedID = new Mat(img, new Rect(topLeft, bottomRight)); 
		
        Mat roi = detectIDMRZ(img);
        
        displayImage(toBufferedImage(img), "found MRZ?");
    	
        // Save the visualized detection.
        System.out.println("Writing "+ outFile);
        Imgcodecs.imwrite(outFile, img);
        Imgcodecs.imwrite("/home/chris/Documents/mrz_roi.jpg", roi);
    }
    
    public static void main(String[] args) {
    	System.out.println(args[0]);
    	
        try{
    	new MatchingDemo().run(args[0], args[1]);
        }
        catch(Exception e)
        {
        	e.printStackTrace();
        }
    }
}

