default: all
.PHONY: default
	
all:
	nvcc -arch=sm_20 -o mat_inv mat_inv.cu -lcublas -lm
.PHONY: all

user: 
	nvcc -arch=sm_20 -o user_mat_inv mat_inv_userInput.cu -lcublas -lm

clean:
	rm mat_inv
.PHONY: all
