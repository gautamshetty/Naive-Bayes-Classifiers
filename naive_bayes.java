
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

public class naive_bayes {

	public static void main(String[] args) {
		
		long st = System.currentTimeMillis();

		String trainingFile = args[0];
		String testFile = args[1];
		String methodType = args[2];
		int numParam = (!methodType.equals("gaussians") ? Integer.valueOf(args[3]) : -1);
		
		if (trainingFile == null || trainingFile.isEmpty() || testFile == null || testFile.isEmpty() || methodType == null || methodType.isEmpty()) {
			System.out.println("Insufficient parameters");
			System.exit(0);
		}
		
		Map trainingData = getData(trainingFile);
		
		List testData = getTestData(testFile);
		
		if (methodType != null && methodType.equals("histograms"))
			histogram(trainingData, numParam, testData);
		else if (methodType != null && methodType.equals("gaussians"))
			gaussians(trainingData, testData);
		else if (methodType != null && methodType.equals("mixtures"))
			mixtureGaussian(trainingData, numParam, testData);
		
		long et = System.currentTimeMillis();
		
		System.out.format("\nTime Required : %.4f %s", (et - st) / 1000.0, " secs");
	}
	
	private static void gaussians(Map data, List testData) {
		
		Map meanMap = new HashMap();
		Map varMap = new HashMap();
		
		calculateMeanAndVarForEachDim(data, meanMap, varMap);
		
		Map testObjects = new TreeMap();
		
		classifyGaussians(data, testData, meanMap, varMap);
	}

	private static void classifyGaussians(Map data, List testData, Map meanMap, Map varMap) {
		
		List testRecord = null;
		String clAttr = null;
		List allClass = new ArrayList();
		List allClassProb = new ArrayList();
		List maxProbList = new ArrayList();
		List maxClass = new ArrayList();
		//List correctClass
		int count = 0, classNum = 0;
		double xTest = 0.0, mean = 0.0, sigma = 0.0, prob = 0.0, classProb = 1.0, maxProb = Double.NEGATIVE_INFINITY;
		double accuracy = 0, classificationAccuracy = 0.0;
		Iterator dataIt = null;
		for (int i = 0; i < testData.size(); i++) {
			
			testRecord = (List) testData.get(i);
			dataIt = data.keySet().iterator();
			classProb = 1.0;
			//for (int j = 0; j < testRecord.size() - 1; j++) {
			count = 0;
			while (dataIt.hasNext()) {
				
				clAttr = (String) dataIt.next();
				mean = (Double) meanMap.get(clAttr);
				sigma = (Double) varMap.get(clAttr);
				
				xTest = Double.valueOf((String) testRecord.get(count));
				
				prob = getGaussian(sigma, mean, xTest);
				
				classProb *= prob; 
				
				count++;
				if (count == testRecord.size() - 1) {
					allClassProb.add(classProb);
					allClass.add(Integer.valueOf(clAttr.split("-")[0]) + 1);
					count = 0;
					classProb = 1.0;
				}
			}
			
			maxProb = (Double) allClassProb.get(0);
			for (int k = 1; k < allClassProb.size(); k++) {
				if ((Double) allClassProb.get(k) > maxProb) {
					maxProb = (Double) allClassProb.get(k); 
				}
			}
			
			for (int k = 0; k < allClassProb.size(); k++) {
				if (maxProb == (Double) allClassProb.get(k)) {
					maxClass.add(allClass.get(k));
				}
			}
			
			if (maxClass != null && !maxClass.isEmpty() && maxClass.contains(Integer.valueOf((String)testRecord.get(testRecord.size() - 1)))) {
				accuracy = 1 / maxClass.size();
				//classProb = maxProb;//(Double) allClassProb.get();
				classNum = (Integer) maxClass.get(maxClass.indexOf(Integer.valueOf((String)testRecord.get(testRecord.size() - 1))));
			}
						
			System.out.printf("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n",
                    i, classNum, maxProb, Integer.valueOf((String)testRecord.get(testRecord.size() - 1)), accuracy);
			
			classificationAccuracy += accuracy;
			
			allClass.clear();
			allClassProb.clear();
			maxClass.clear();
			maxProb = Double.NEGATIVE_INFINITY;
			accuracy = 0;
		}
		
		classificationAccuracy /= testData.size();
		
		System.out.printf("\nclassification accuracy=%6.4f\n", classificationAccuracy);
	}
	
	private static void classifyMixtureGaussians(Map data, List testData, Map meanMap, Map varMap, Map weights, int noOfGaussians) {
		
		List testRecord = null;
		String clAttr = null;
		List allClass = new ArrayList();
		List allClassProb = new ArrayList();
		List maxClass = new ArrayList();
		
		int count = 0, classNum = 0;
		double xTest = 0.0, mean = 0.0, sigma = 0.0, prob = 0.0, sumProb = 0.0, classProb = 1.0, maxProb = Double.NEGATIVE_INFINITY, wgt = 0.0;
		double accuracy = 0, classificationAccuracy = 0.0;
		Iterator dataIt = null;
		for (int i = 0; i < testData.size(); i++) {
			
			testRecord = (List) testData.get(i);
			dataIt = data.keySet().iterator();
			classProb = 1.0;
			count = 0;
			while (dataIt.hasNext()) {
				
				clAttr = (String) dataIt.next();
				sumProb = 0.0;
				for (int k = 0; k < noOfGaussians; k++) {
					
					mean = (Double) meanMap.get(clAttr+"-"+(k + 1));
					sigma = (Double) varMap.get(clAttr+"-"+(k + 1));
					wgt = (Double) weights.get(clAttr+"-"+(k + 1));
					
					xTest = Double.valueOf((String) testRecord.get(count));
					
					prob = getGaussian(sigma, mean, xTest);
					
					sumProb += prob * wgt;
				}
				
				classProb *= sumProb;
				
				count++;
				if (count == testRecord.size() - 1) {
					allClassProb.add(classProb);
					allClass.add(Integer.valueOf(clAttr.split("-")[0]) + 1);
					count = 0;
					classProb = 1.0;
				}
			}
			
			if (!allClassProb.isEmpty()) {
				maxProb = (Double) allClassProb.get(0);
				for (int k = 1; k < allClassProb.size(); k++) {
					if ((Double) allClassProb.get(k) > maxProb) {
						maxProb = (Double) allClassProb.get(k); 
					}
				}
				
				for (int k = 0; k < allClassProb.size(); k++) {
					if (maxProb == (Double) allClassProb.get(k)) {
						maxClass.add(allClass.get(k));
					}
				}
			}
			
			if (maxClass != null && !maxClass.isEmpty() && maxClass.contains(Integer.valueOf((String)testRecord.get(testRecord.size() - 1)))) {
				accuracy = 1 / maxClass.size();
				//classProb = maxProb;//(Double) allClassProb.get();
				classNum = (Integer) maxClass.get(maxClass.indexOf(Integer.valueOf((String)testRecord.get(testRecord.size() - 1))));
			}
			
			System.out.printf("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n",
                    i, classNum, maxProb, Integer.valueOf((String)testRecord.get(testRecord.size() - 1)), accuracy);
			
			classificationAccuracy += accuracy;
			
			allClass.clear();
			allClassProb.clear();
			maxClass.clear();
			maxProb = Double.NEGATIVE_INFINITY;
			accuracy = 0;
		}
		
		classificationAccuracy /= testData.size();
		
		System.out.printf("\nclassification accuracy=%6.4f\n", classificationAccuracy);
	}
	
	private static Map calculateMeanAndVarForEachDim(Map data, Map meanMap, Map varMap) {
		
		String clAttr = null;
		List attrList = null;
		double value = 0.0;
		String [] clAttrLabel = null;
		Iterator clAttrIt = data.keySet().iterator();
		while (clAttrIt.hasNext()) {
			
			clAttr = (String) clAttrIt.next();
			attrList = (List) data.get(clAttr);
			value = 0.0;
			for (int i = 0; i < attrList.size(); i++) {
				value += (double) attrList.get(i);
			}
			
			value = value / attrList.size();
			meanMap.put(clAttr, value);
			
			value = 0.0;
			for (int i = 0; i < attrList.size(); i++) {
				value += Math.pow((double) attrList.get(i) - (double) meanMap.get(clAttr), 2);
			}
			
			value = ((attrList.size() - 1) == 0 ? 0 :Math.sqrt(value / (attrList.size() - 1)));
			if (value < 0.01)
				value = 0.01;
			varMap.put(clAttr, value);
			
			clAttrLabel = clAttr.split("-");
			System.out.format("Class %d, attribute %d, mean = %.2f, std = %.2f\n", 
									Integer.valueOf(clAttrLabel[0]) + 1, Integer.valueOf(clAttrLabel[1]),
								   	(Double) meanMap.get(clAttr), (Double) varMap.get(clAttr));
		}
		
		return meanMap;
	}
	
	private static void mixtureGaussian(Map data, int noOfGaussians, List testData) {
		
		Map meanAttr = new HashMap();
		Map stdDev = new HashMap();
		Map weights = new HashMap();
		
		initialize(data, meanAttr, stdDev, weights, noOfGaussians);
		
		for (int count = 0; count < 50; count++) {
			
			Map probMap = calculateProbability(data, noOfGaussians, meanAttr, stdDev, weights);
			
			updateParameters(data, noOfGaussians, meanAttr, stdDev, probMap, weights);
		}
		
		printValues(meanAttr, stdDev);
		
		classifyMixtureGaussians(data, testData, meanAttr, stdDev, weights, noOfGaussians);
	}
	
	private static void printValues(Map meanAttr, Map stdDev) {
		
		String gausClAttr = null;
		String [] clAttrGaus = null;
		double mean = 0.0, sig = 0.0;
		
		Set meanSet = new TreeSet(meanAttr.keySet());
		Iterator meanIt = meanSet.iterator();
		while (meanIt.hasNext()) {
			
			gausClAttr = (String) meanIt.next();
			mean = (Double) meanAttr.get(gausClAttr);
			sig = (Double) stdDev.get(gausClAttr);
			
			clAttrGaus = gausClAttr.split("-");
			System.out.format("Class %d, attribute %d, Gaussian %d, mean = %.2f, std = %.2f\n", 
								Integer.valueOf(clAttrGaus[0]) + 1, Integer.valueOf(clAttrGaus[1]), Integer.valueOf(clAttrGaus[2]), mean, sig);
		}
	}
	
	private static void updateParameters(Map data, int noOfGaussians, Map meanAttr, Map stdDev, Map probMap, Map weights) {
		
		meanAttr.clear();
		stdDev.clear();
		weights.clear();
		
		String clAttr = null;
		List attrList = null, probList = null;
		double pijxj = 0.0, sumPij = 0.0, mean = 0.0, sig = 0.0, allGausProbij = 0.0, w = 0.0;
		
		Iterator dataIt = data.keySet().iterator();
		while (dataIt.hasNext()) {
			clAttr = (String) dataIt.next();
			attrList = (List) data.get(clAttr);
			
			for (int k = 0; k < noOfGaussians; k++) {
				
				probList = (List) probMap.get(clAttr+"-"+(k + 1));
				for (int j = 0; j < probList.size(); j++)
					sumPij += (Double) probList.get(j);
				
				for (int j = 0; j < attrList.size(); j++)
					pijxj += (Double) probList.get(j) * (Double) attrList.get(j);
				
				mean = pijxj / sumPij;
				meanAttr.put(clAttr+"-"+(k + 1), mean);
				
				for (int j = 0; j < attrList.size(); j++) 
					sig +=  (Double) probList.get(j) * Math.pow(((Double) attrList.get(j) - (Double) meanAttr.get(clAttr+"-"+(k + 1))), 2);
				
				sig = sig / sumPij;
				sig = Math.sqrt(sig);
				if (sig < 0.01)
					sig = 0.01;
				stdDev.put(clAttr+"-"+(k + 1), sig);
				
				for (int kIndex = 0; kIndex < noOfGaussians; kIndex++) {
					probList = (List) probMap.get(clAttr+"-"+(kIndex+1));
					for (int j = 0; j < probList.size(); j++)
						allGausProbij += (Double) probList.get(j);
				}
				w = sumPij / allGausProbij;
				weights.put(clAttr+"-"+(k + 1), w);
			}
		}
	}
	
	private static Map calculateProbability(Map data, int noOfGaussians, Map meanAttr, Map stdDev, Map weights) {
		
		List attrList = null, probList = null;
		String clAttr = null;
		double nixj = 0.0, sig = 0.0, m = 0.0, x = 0.0, normFact = 0.0, w = 0.0, probij = 0.0;
		Map probMap = new HashMap(); 
		
		Iterator dataIt = data.keySet().iterator();
		while (dataIt.hasNext()) {
			
			clAttr = (String) dataIt.next();
			attrList = (List) data.get(clAttr);
			for (int k = 0; k < noOfGaussians; k++) {
				for (int j = 0; j < attrList.size(); j++) {
					
					x = (Double) attrList.get(j);
					m = (Double) meanAttr.get(clAttr+"-"+(k + 1));
					sig = (Double) stdDev.get(clAttr+"-"+(k + 1));
					w = (Double) weights.get(clAttr+"-"+(k + 1));
					
					nixj = getGaussian(sig, m, x) * w;
					
					normFact = normalizationFactor(x, w, clAttr, meanAttr, stdDev, noOfGaussians);
					
					probij = nixj * w / normFact;
					
					probList = (List) probMap.get(clAttr+"-"+(k + 1));
					if (probList == null) {
						probList = new ArrayList();
						probMap.put(clAttr+"-"+(k + 1), probList);
					}
					probList.add(probij);
				}
			}
		}
		
		return probMap;
	}
	
	private static double getGaussian(double sig, double m, double x) {
		
		double ePow = - (Math.pow((x - m), 2) / (2 * Math.pow(sig, 2)));
		return (1/(sig * Math.sqrt(2 * Math.PI))) * Math.exp(ePow);
	}
	
	private static double normalizationFactor(double x, double w, String clAttr, Map meanAttr, Map stdDev, int noOfGaussians) {
		
		double normFact = 0.0, m = 0.0, sig = 0.0, nixj = 0.0;
		for (int k = 0; k < noOfGaussians; k++) {
			
			m = (Double) meanAttr.get(clAttr+"-"+(k + 1));
			sig = (Double) stdDev.get(clAttr+"-"+(k + 1));
			
			nixj = getGaussian(sig, m, x);
			
			normFact += w * nixj;
		}
		
		if (normFact == 0)
			System.out.println();
		
		return normFact;
	}
	
	private static void initialize(Map data, Map meanAttr, Map stdDev, Map weights, int noOfGaussians) {
		
		List attrList = null;
		String clAttr = null;
		double g = 0.0;//, mean = 0.0;
		//List wgtList = null;
		
		Iterator dataIt = data.keySet().iterator();
		while (dataIt.hasNext()) {
			
			clAttr = (String) dataIt.next();
			attrList = (List) data.get(clAttr);
			
			Collections.sort(attrList);
			g = ((Double) attrList.get(attrList.size() - 1) - (Double) attrList.get(0)) / noOfGaussians;
					
			//Mean and Standard Deviation.
			for (int k = 0; k < noOfGaussians; k++) {
				//meanAttr.put((k + 1)+"-"+clAttr, ((((Double) attrList.get(0) + (k + 1) * g) / 2) + mean) / attrList.size());
				meanAttr.put(clAttr+"-"+(k + 1), ((Double) attrList.get(0) + (k + 1) * g) / 2);
				
				stdDev.put(clAttr+"-"+(k + 1), (double) 1);
				
				weights.put(clAttr+"-"+(k + 1), ((double)1/noOfGaussians));
			}
		}
	}
	
	private static void histogram (Map trainingData, int noOfBin, List testData) {
		
		Map binWidthMap = new HashMap();
		Map allocBinMap = new HashMap();
		
		Map binAttrMap = allocateBin(trainingData, binWidthMap, allocBinMap, noOfBin);
		
		calculateProbability(binAttrMap, binWidthMap);
		
		String clAttr = null;
		List testRecord = null, attrList = null;
		List allClass = new ArrayList();
		List allClassProb = new ArrayList();
		List maxClass = new ArrayList();
		
		Iterator dataIt = null;
		Set allocBinList = null;
		int count = 0, classNum = 0, binNum = 0;
		double accuracy = 0, classificationAccuracy = 0.0;
		double xTest = 0.0, prob = 0.0, sumProb = 0.0, classProb = 1.0, maxProb = 0.0;
		Iterator binNumIt = null;
		for (int i = 0; i < testData.size(); i++) {
			
			testRecord = (List) testData.get(i);
			dataIt = trainingData.keySet().iterator();
			classProb = 1.0;
			count = 0;
			while (dataIt.hasNext()) {
				
				clAttr = (String) dataIt.next();
				allocBinList = (Set) allocBinMap.get(clAttr);
				sumProb = 0.0;
				//for (int k = 0; k < allocBinList.size(); k++) {
				binNumIt = allocBinList.iterator();
				while(binNumIt.hasNext()) {	
					binNum = (Integer) binNumIt.next();
					xTest = Double.valueOf((String) testRecord.get(count));
					
					attrList = (List) binAttrMap.get(clAttr+"-"+binNum);
					
					prob = (double)getAttrCountInBin(xTest, attrList) / attrList.size();
					
					sumProb += prob;
				}
				
				classProb *= sumProb;
				
				count++;
				if (count == testRecord.size() - 1) {
					allClassProb.add(classProb);
					allClass.add(Integer.valueOf(clAttr.split("-")[0]) + 1);
					count = 0;
					classProb = 1.0;
				}
			}
			
			maxProb = (Double) allClassProb.get(0);
			for (int k = 1; k < allClassProb.size(); k++) {
				if ((Double) allClassProb.get(k) > maxProb) {
					maxProb = (Double) allClassProb.get(k); 
				}
			}
			
			for (int k = 0; k < allClassProb.size(); k++) {
				if (maxProb == (Double) allClassProb.get(k)) {
					maxClass.add(allClass.get(k));
				}
			}
			
			if (maxClass != null && !maxClass.isEmpty() && maxClass.contains(Integer.valueOf((String)testRecord.get(testRecord.size() - 1)))) {
				accuracy = 1 / maxClass.size();
				//classProb = maxProb;//(Double) allClassProb.get();
				classNum = (Integer) maxClass.get(maxClass.indexOf(Integer.valueOf((String)testRecord.get(testRecord.size() - 1))));
			}
			
			System.out.printf("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f\n",
                    i, classNum, maxProb, Integer.valueOf((String)testRecord.get(testRecord.size() - 1)), accuracy);
			
			classificationAccuracy += accuracy;
			
			allClass.clear();
			allClassProb.clear();
			maxClass.clear();
			//maxProb = Double.NEGATIVE_INFINITY;
			accuracy = 0;
		}
		
		classificationAccuracy /= testData.size();
		
		System.out.printf("\nclassification accuracy=%6.4f\n", classificationAccuracy);
	}

	private static void calculateProbability(Map trainingData, Map binWidthMap) {
		
		String attrIndex = null;
		List attrList = null;
		double attr = 0.0, prob = 1.0;
		String [] rowArray = null;
		double binWidth = 1;
		
		Iterator attrNumIt = trainingData.keySet().iterator();
		while (attrNumIt.hasNext()) {
			attrIndex = (String) attrNumIt.next();
			attrList = (List) trainingData.get(attrIndex);
			binWidth = (Double) binWidthMap.get(attrIndex);
			
			prob = 1.0;
			for (int i = 0; i < attrList.size(); i++) {
				attr = (Double) attrList.get(i);
				prob *= ((double) getAttrCountInBin(attr, attrList) / (attrList.size() * binWidth));
			}
			
			rowArray = attrIndex.split("-");
			System.out.format("Class %d, attribute %d, bin %d, P(bin | class) = %.2f\n", Integer.valueOf(rowArray[0]) + 1, 
									Integer.valueOf(rowArray[1]), Integer.valueOf(rowArray[2]), prob);
		}
	}
	
	private static int getAttrCountInBin(double attr, List attrList) {
		
		int count = 0;
		
		for (int i = 0; i < attrList.size(); i++) {
			if (attr == (double) attrList.get(i))
				count++;
		}
		
		return count;
	}

	private static Map allocateBin(Map attrData, Map binWidth, Map allocBin, int numOfBin) {
		
		List attrList = null, binAttrList = null;
		Set allocBinList = null;
		double l = 0.0, s = 0.0, g = 0.0, maxBinValue = 0.0, attr = 0.0, minBinValue = 0.0;
		int binNum = 0;
		Map binAttrMap = new TreeMap();
		String attrIndex = null;
		//List binWidth = new ArrayList();
		
		Iterator attrNumIt = attrData.keySet().iterator();
		while (attrNumIt.hasNext()) {
			
			attrIndex = (String) attrNumIt.next();
			//System.out.println("attr Index : "+attrIndex);
			attrList = (List) attrData.get(attrIndex);
			Collections.sort(attrList);
			
			//System.out.println("maxBinValue : "+maxBinValue);
			//binAttrMap = new HashMap();
			for (int i = 0; i < attrList.size(); i++) {//attrList.size(); 
				
				s = (Double) attrList.get(0);
				l = (Double) attrList.get(attrList.size() - 1);
				
				g = (l - s) / (numOfBin - 3);
				
				//System.out.println("Class-Attr : "+attrIndex+" S : "+s+" L : "+l+" G : "+g);
				
				binNum = 0;
				minBinValue = Double.NEGATIVE_INFINITY;
				maxBinValue = s - (g / 2);
//				System.out.println(" Min Value : "+minBinValue+" Max Value : "+maxBinValue);
				
				//binWidth.add(Double.POSITIVE_INFINITY);
				attr = (Double) attrList.get(i);
				int count = 0;
				//System.out.println("attr : "+attr+" max Bin : "+maxBinValue+" bin : "+binNum);
				while (true) {
					
					//System.out.println("Count : "+ count++ +" Bin : "+binNum);
					if ((attr >= minBinValue && attr < maxBinValue) || binNum == (numOfBin - 1)) {
//						System.out.println("Attr : "+attr+" Min Value : "+minBinValue+" Max Value : "+maxBinValue);
						break;
					}
					
					binNum += 1;
					s += (g/2);
					minBinValue = maxBinValue;
					maxBinValue = (binNum == numOfBin - 1 ? Double.POSITIVE_INFINITY : s);
//					System.out.println(" Min Value : "+minBinValue+" Max Value : "+maxBinValue);
				}
				
				allocBinList = (Set) allocBin.get(attrIndex);
				if (allocBinList == null) {
					allocBinList = new HashSet();
					allocBin.put(attrIndex, allocBinList);
				}
				allocBinList.add(binNum);
				
				if (minBinValue == Double.NEGATIVE_INFINITY || maxBinValue == Double.POSITIVE_INFINITY)
					binWidth.put(attrIndex+"-"+binNum, Double.POSITIVE_INFINITY);
				else 
					binWidth.put(attrIndex+"-"+binNum, g);
				
				binAttrList = (List) binAttrMap.get(attrIndex+"-"+binNum);
				if (binAttrList == null) {
					binAttrList = new ArrayList();
					binAttrMap.put(attrIndex+"-"+binNum, binAttrList);
				}
				
				binAttrList.add(attr);
			}
			
			//attrData.put(attrIndex, binAttrMap);
		}
		return binAttrMap;
	}
	
	private static Map getData(String fileName) {
		
		BufferedReader bs = null;
		FileReader fr = null;
		Map attrData = new TreeMap();
		
		try {
			
			fr = new FileReader(fileName);
			bs = new BufferedReader(fr);
			
			String [] rowArray = null;
			String str = null;
			
			int attrCount = 0, rowCount = 0;;
			List attrList = null;
			while (true) {
				
				str = bs.readLine();
				if (str == null)
					break;
				
				rowArray = str.split("\\s+");
				
				boolean skip = false;
				for (int i = 0; i < rowArray.length - 2; i++) {
					if (rowArray[i] != null && !rowArray[i].isEmpty() && rowArray[i] != "") {
						if (Double.valueOf(rowArray[i]) > 0.75) {
							skip = true;
							rowCount += 1;
							//System.out.println("Row : "+rowCount);
						}
					}
				}
				
				if (skip)
					continue;
				
				attrCount = 0;
				for (int i = 0; i < rowArray.length - 1; i++) {
					
					if (rowArray[i] != null && !rowArray[i].isEmpty() && rowArray[i] != "") {
						
						attrList = (List) attrData.get((Integer.valueOf(rowArray[rowArray.length - 1]) - 1)+"-"+Integer.valueOf(attrCount));
						if (attrList == null) {
							attrList = new ArrayList();
							attrData.put((Integer.valueOf(rowArray[rowArray.length - 1]) - 1)+"-"+Integer.valueOf(attrCount), attrList);
						}
						attrList.add(Double.valueOf(rowArray[i]));
						attrCount++;
						//System.out.println("Key : "+rowArray[rowArray.length - 1]+"-"+Integer.valueOf(i) + " value : "+rowArray[i]);
					}
				}
				
			}
			
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		
		return attrData;
	}
	
	private static List getTestData(String fileName) {
		
		BufferedReader bs = null;
		FileReader fr = null;
		//Map clData = new TreeMap();
		List testRecords = new ArrayList();
		
		try {
			
			fr = new FileReader(fileName);
			bs = new BufferedReader(fr);
			
			String [] rowArray = null;
			List rowList = null;
			String str = null;
			int attrCount = 0;
			List classVal = null;
			while (true) {
				
				str = bs.readLine();
				if (str == null)
					break;
				
				rowArray = str.split("\\s+");
				rowList = new ArrayList();
				attrCount = 0;
				for (int i = 0; i < rowArray.length; i++) {
					if (rowArray[i] != null && !rowArray[i].isEmpty() && rowArray[i] != "")
						rowList.add(rowArray[attrCount]);
					attrCount++;
				}
								
				testRecords.add(rowList);				
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		
		return testRecords;
	}
}
