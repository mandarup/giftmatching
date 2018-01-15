@everywhere using ProgressMeter

# function test_parallel()
#     a = SharedArray{Float64}((100,2), init=0)
#     @parallel for i = 1:10
#         ind = rand(1:100,10)
#         a[ind,2] = 1
#         #@show a[i]
#     end
#     return a
# end


# @everywhere function test()
#     a = SharedArray{Float64}((100,2), init=0)
#     @showprogress for i = 1:10000000
#         ind = rand(1:100,10)
#         a[ind,2] = rand(1:100,10)
#     end
#     println(a)
# end


# @everywhere function test()
#     a = SharedArray{Float64}((100,2), init=0)
#     tic()
#     @parallelprogress for i = 1:10000000
#         ind = rand(1:100,10)
#         a[ind,2] = rand(1:100,10)
#     end
#     print(toc())
#     println(a)
# end

function test_par()
    a = SharedArray{Float64}((100,2), init=0)
    tic()
    @parallel for i = 1:10000000
        ind = rand(1:100,10)
        a[ind,2] = rand(1:100,10)
    end
    sleep(.1)
    print(toc())
    println(a)
end


function test()
    a = Array{Float64}(100,2)
    tic()
    for i = 1:10000000
        ind = rand(1:100,10)
        a[ind,2] = rand(1:100,10)
    end
    sleep(.1)
    print(toc())
    println(a)
end


addprocs(6)

@everywhere function test_par2()
    a = SharedArray{Float64,2}((100,2), init=0)
    tic()
    @parallel for i = 1:10000000
        ind = rand(1:100,10)
        a[ind,2] = rand(1:100,10)
    end


    sleep(.1)
    print(toc())
    println(a)
end




test()
test_par()

test_par2()



@everywhere function test_par3()
    a = SharedArray{Float64,1}((1), init=0)
    tic()
    @parallel for i = 1:10000000
        a[1] += 1
    end

    a = @parallel (+) for i = 1:100000
        1
    end
    sleep(.1)
    print(toc())
    println(a)
end

@everywhere function test_par4()
    tic()
    a = @parallel (+) for i = 1:100000
        1
    end

    print(toc())
    println(a)
    @show a
end

test_par3()
test_par4()

nheads = @parallel (+) for i = 1:200000000
    Int(rand(Bool))
end

@time a = @parallel (append!) for i = 1:100000000
    [1]
end

a = []
@time for i = 1:100000000
    append!(a, [1])
end
