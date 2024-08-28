CC        = mpicc
HYPRE_DIR = /usr/local/hypre

# compile configs
CFLAGS = -g -Wall -I$(HYPRE_DIR)/include
# link configs
LFLAGS = -L$(HYPRE_DIR)/lib -lHYPRE -lm

# 当make发现.o目标不存在或者老的时候，自动运行以下命令. Note: $< 表示所依赖的第一个文件名xxx.c
# %.o : %.c
# 	$(CC) $(CFLAGS) -c $<
# 	@echo "Compiled $< to $@"
# 目标:hello, 依赖文件: hello_world.o, $@: 代指目标, $^表示所有的依赖文件



ij: ij.c
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)
mysolvertest: mysolver.c
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)
mysolver: mysolver.c
	$(CC) -O3 -o $@ $^ $(CFLAGS) $(LFLAGS)
