using LinearAlgebra

# 1. Modifiez la fonction suivante pour qu'elle renvoie la solution x
#    du système triangulaire supérieur Rx = b.
#    Votre fonction ne doit modifier ni R ni b.
function backsolve(R::UpperTriangular, b)
    x = similar(b)
    n = length(b)                     
    for i in n:-1:1                    
        s = zero(eltype(b))            
        for j in i+1:n                 
            s += R[i,j] * x[j]         
        end
        x[i] = (b[i] - s) / R[i,i]     
    end
    return x
end

# 2. Modifiez la fonction suivante pour qu'elle renvoie la solution x
#    du système Hessenberg supérieur Hx = b ou du problème aux moindres
#    carrés min ‖Hx - b‖ à l'aide de rotations ou
#    réflexions de Givens et d'une remontée triangulaire.
#    Votre fonction peut modifier H et b si nécessaire.
#    Il n'est pas nécessaire de garder les rotations en mémoire et la
#    fonction ne doit pas les renvoyer.
#    Seul le cas réel sera testé ; pas le cas complexe.
function hessenberg_solve(H::UpperHessenberg, b)
    m, n = size(H)
    for j = 1:n
        if j < m
            e1, e2 = H[j, j], H[j+1, j]
            r = sqrt(e1^2 + e2^2) # norme euclidienne
            if r != 0
                c = e1 / r
                s = -e2 / r
                # Applique la rotation à H
                for k = j:n
                    e1, e2 = H[j, k], H[j+1, k]
                    H[j, k]     =  c * e1 - s * e2
                    H[j+1, k]   =  s * e1 + c * e2
                end
                # Applique la rotation à b
                e1, e2 = b[j], b[j+1]
                b[j]   =  c * e1 - s * e2
                b[j+1] =  s * e1 + c * e2
            end
        end
    end
    # Après les rotations, les n premières lignes de H forment une matrice triangulaire supérieure
    R = UpperTriangular(H[1:n, 1:n])
    x = backsolve(R, b[1:n])
    return x
end

# vérification
using Test
@testset "Tests systèmes triangulaires et Hessenberg" begin
    for n ∈ (10, 20, 30)
        # square system
        A = rand(n, n)
        A[diagind(A)] .+= 1
        b = rand(n)
        R = UpperTriangular(A)
        x = backsolve(R, b)
        @test norm(R * x - b) ≤ sqrt(eps()) * norm(b)
        H = UpperHessenberg(A)
        x = hessenberg_solve(copy(H), copy(b))
        @test norm(H * x - b) ≤ sqrt(eps()) * norm(b)
        # slightly overdetermined least squares
        A = rand(n + 1, n)
        A[diagind(A)] .+= 1
        H = UpperHessenberg(A)
        b = rand(n + 1)
        x_ls = hessenberg_solve(copy(H), copy(b))
        x_qr = H \ b
        @test norm(x_ls - x_qr) ≤ sqrt(eps()) * norm(x_qr)
    end
end