def remove_dups(lista):
    if [] == lista:
        return lista
    elif lista[-1] in lista[:-1]:
        return remove_dups(lista[:-1])
    else:
        return remove_dups(lista[:-1]) + [lista[-1]]
    

#print(remove_dups([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]))

#lista = [1]

#lista = lista + [2]

#print(lista)




#chave secreta = [1, 0, 0, 1, 0]




#ultima_tentativa = [0, 0, 0, 0, 0]
#incorretos = 2  x = 2
#tentativas = 0


#ultima_tentativa = [1, 0, 0, 0, 0]
#incorretos = 1  x = 2.... x = 1
#tentativas = 1

#ultima_tentativa = [1, 1, 0, 0, 0]
#incorretos = 2 .. x = 1
#tentativas = 2


#ultima_tentativa = [1, 0, 1, 0, 0]
#incorretos = 2 .. x = 1
#tentativas = 3

#ultima_tentativa = [1, 0, 0, 1, 0]
#incorretos = 0 .. x = 1 .. x = 0
#tentativas = 4


s1 = "123456789"
s2 = "987654321"

#print(s1[::-1] == s2)

lista = [1, 5, 7, 2, 4, 8, 1, 13]

#print(sorted(lista)[-2])

lista = ["-"] * 10

print(lista)
