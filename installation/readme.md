# Installation of the modelling environment
We use TensorFlow and Keras for this code. We will create a new virtual enviroment for modelling; in this way, we can install all the required modules without interference from (or without itself interfering) other Python-based packages you may use.
Please follow these steps to reproduce our (virtual) environment:
1. Install Anaconda (anaconda.com)
2. Download your yml file (for Mac or for Windows). One way to do this, is by navigating to the main branch of this directory, click Code (the green button), and then download zip. You then have a zip file with all code, also the yml files. Another (equally valid) way to obtain the yml file is to simply copy-paste the content from the yml file you need (for Mac or for Windows), and then save this content as a local yml file on your own machine. 
3. On the anaconda prompt on Windows ([how to find it?](https://www.youtube.com/watch?v=UAUO_K-bRMs)), or on the terminal prompt on Mac (typically located in your icons bar), navigate to the location where your yml file is downloaded. You can navigate to a location by typing at the terminal prompt (e.g.) cd /Users/tom/Downloads (on Mac) or cd C:\Users\tom\Downloads (on Windows). (The command cd is short for "change directory", so it brings you to that directory.) 
4. On the anaconda prompt (terminal prompt on Mac), type
```conda env create --file [name of yml file]```
For example, if your file is called modelling_mac.yml, then you should type ```conda env create --file modelling_mac.yml```

After installation, a new virtual environment should now have been created called modelling. 

You can check the list of available environments by typing ```conda env list``` at the prompt. If installation went well, you should see modelling in that list.

You can activate the environment you want by typing ```conda activate [name of env]``` at the prompt. For example, to activate environment modelling, type ```conda activate modelling```. You can then start the code editor we will use, which is Spyder: Just type spyder at the prompt. Spyder will then start, and you're ready to go.

Learn how to navigate Anaconda with its [cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).
