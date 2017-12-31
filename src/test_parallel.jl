function test_parallel()
    a = SharedArray{Float64}((1000000,1000000), init=0)
    @parallel for i = 1:1000000
        for j = 1:1000000
            a[i,j] += 1
            #@show a[i]
        end
    end
    return a
end
#
a = test_parallel()
# print(a)

println(a[1:100])
