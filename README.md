# To Do


- ProteinMPNN starting to be implemented but not yet completed
- BACKGROUND_JOB starting to be implemented but not yet completed
- Add penalty if design results in a sequence that already exists? 


<blockquote>
  <strong>@Abbie:</strong>
  
  Please adjust ./design/AIzyme_Run_Abbie_1.ipynb so that the paths point to the correct files, this should be:

  For AIzyme_Functions: ./src/AIzyme_Functions_009.ipynb
 
  for the parent folder: ./design/Parent_1ohp
</blockquote>


# AIzymes
This will be the AIzymes Github. Lets use the readme to collect notes.

> [!IMPORTANT]
> Track all versions of AIzyme_Functions in ./src/README.md

> [!TIP]
> Read [AIzymes Manual.docx](https://github.com/bunzela/AIzymes/blob/main/AIzymes%20Manual.docx) for detailed information on AIzymes
> 
> Feel free to change the Manual, but always track changes!

> [!TIP]
> See [How_to_push_and_pull.txt](https://github.com/bunzela/AIzymes/blob/main/How_to_push_and_pull.txt) for a quick manual how to push and pull changes to GitHub.

# Requirments
AIzymes requires ESMfold through HuggingFace (https://huggingface.co/facebook/esmfold_v1)
```
pip install --upgrade transformers py3Dmol accelerate
```
