/*
 * Macro template to process multiple images in a folder
 */

#@ File (label = "Input directory", style = "directory") input
#@ File (label = "Output directory", style = "directory") output
#@ String (label = "File suffix", value = ".tif") suffix

// See also Process_Folder.py for a version of this code
// in the Python scripting language.

processFolder(input);

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i]))
			processFolder(input + File.separator + list[i]);
		if(endsWith(list[i], suffix))
			processFile(input, output, list[i]);
	}
}

function processFile(input, output, file) {
	// Do the processing here by adding your own code.	
	// print("Processing: " + input + File.separator + file);
	emb="emb1";

	s=9;
	n=5; 
	surfix_index = indexOf(file, ".");
	prefix = output + File.separator + substring(file, 0, surfix_index);
	for (i=s; i<s+n; i++)         
	{ 
		 open(input + File.separator + file);
		 setSlice(i);
		 wait(50);
		 r1 = random();
		 r2 = random();
		 makeRectangle(160 + (249 - 128) * r1, 339 + (173 - 128) * r2, 128, 128);
		 run("Crop");
		 run("Median 3D...", "x=2 y=2 z=2");
		 setMinAndMax(2500, 4200);

	     a=IJ.pad(i, 2); 
	     // showProgress(i, n); 

	     saveAs("Jpeg", prefix + "_" + emb + "_slice" + a + ".jpg"); 
	     close();
	} 	
}
