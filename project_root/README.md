# FraudSim – AI Scam Simulation Generator

A multi-module AI tool for generating simulated fraud content (text, voice, visual posters, warning cards) to support educational/publicity scenarios. Support GPU priority operation and automatically switch to CPU when there is no GPU.

##  Project structure

```
project_root/
│
├── main_controller_text.py   # Main control program with text generation
├── main_controller.py        # Main control program without text generation
├── deepseek_final.py         # Text generator
├── audio.py                  # Text to speech module
├── poster.py                 # Text to poster module
├── banner.py                 # Text to banner module
│
├── text_prompt/              # JSON file generated from text
├── selected_text_prompt/     # JSON files processed and archived 
├── outputs/
│   ├── audio/                
│   ├── poster/            
│   └── banner/ 
├── stickers/                 # Sticker for banner and poster
│
├──persona_seed.json
├──scam_type_seed.json
│
├── requirements.txt          # Dependency List
└── README.md                 # Instructions
```


---

## Set up

### 1. Environment-dependent installation

* Create and activate a new Conda environment for this project or use the one you prefer.
  ```
  conda create -n fraudsim python=3.10
  conda activate fraudsim
  ```
* Install project dependencies
  ```
  pip install -r requirements.txt
  ```
### 2. Deploy Ollama  
* Visit the [official website](https://ollama.com) to download and install the Ollama local deployment tool.  
* After downloading and installing, run the following command in the terminal:
  ```
  ollama run deepseek-r1:8b
  ```

### 3. Run the main program  
```
  python main_controller_text.py
```  
This command will perform the following operations:  
* Identify two types of seeds in the JSON files and run:
  * `deepseek_final.py`：Generate simulated fraud script text.
* Automatically call the corresponding module according to the scenario:  
  * `audio.py`：Generate telephone voice synthesis content.
  * `banner.py`：Generate social media pop-up images.
  * `poster.py`：Generate poster used in email, etc.  
* The final output will be saved to the corresponding folder automatically:  
  * `outputs/audio/`：The generated '.wav 'audio file
  * `outputs/poster/`：The generated poster image
  * `outputs/banner/`：The generated warning card image  
  If the folder does not exist, it will be created automatically.
* The processed script will be moved to `selected_text_prompt/` folder.

**OR**  
```
  python main_controller.py
```  
This command will perform the following operations:  
* Read the generated text scripts JSON file from the `text_prompt/` folder, generate the corresponding outputs in batches, and automatically archive them in the `outputs/` folders as described above.  
* The processed script will be moved to `selected_text_prompt/` folder.


---

