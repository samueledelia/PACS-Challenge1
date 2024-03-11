all:
	g++ -c minimize_function.cpp -fPIC
	g++ -shared minimize_function.o -o libminimize_function.so
	g++ main.cpp -o main -Wl,-rpath = /mnt/c/Users/samue/VSCodeProject/challenge-1/src/libminimize_function.so

clean:
	rm minimize_function.o libminimize_function.so


distclean:
	$(MAKE) clean
	$(RM) -f ./doc $(DEPEND)
	$(RM) *.out *.bak *~
