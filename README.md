# MultiTaskLearning

Overleaf link - https://www.overleaf.com/9841741tdmzmksynxzs 
1) Akash - https://arxiv.org/pdf/1611.01587.pdf, https://arxiv.org/pdf/1611.05377.pdf
2) Ankit - http://www.pnas.org/content/114/13/3521.full.pdf, https://arxiv.org/pdf/1605.06391.pdf 
3) Aniket - https://arxiv.org/pdf/1705.08142.pdf, http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7780802 

Task 1 					Task 2			Task 3
	Horse				Zebra			Dolphin
	Kayak				Canoe			Ketch-101

Alexnet pretrained on Imagenet
Case 1) STL
	Finetune on Task 1/2/3 Labels 0,1 (H, K)
	3 splits (Train, Val, Test)
	Accuracy – Test _____
Case 2) Cross-stitch (MTL)
	Finetune on Task 1 AND Task 2 (High Similarity)
Case 3) Cross-stitch (MTL)
	Finetune on Task 2 AND Task 3 (High Difference)
