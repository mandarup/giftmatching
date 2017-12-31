

function avg_normalized_happiness_parallel(output, child_pref, gift_pref)
    # zip(output[:,1], output[:,2])
    gift_assignment = Dict(zip(output[:,1], output[:,2]))
    gift_counts = Dict(v=>0 for (k,v) in gift_assignment )
    tic()
    for (kid, gift) in gift_assignment
        gift_counts[gift] += 1
        if gift_counts[gift] > N_GIFT_QUANTITY
            throw(AssertionError("quantity for $gift==$(gift_counts[gift]) >=$N_GIFT_QUANTITY"))
        end
    end
    println("couting gifts ",toc())

    max_child_happiness = N_GIFT_PREF * RATIO_CHILD_HAPPINESS
    max_gift_happiness = N_CHILD_PREF * RATIO_GIFT_HAPPINESS


    tic()
    println("computing parallel")
    child_happiness_collect = SharedArray{Float64}(N_CHILDREN, init=0)
    # gift_happiness_collect = SharedArray{Int64}(size(child_pref)[1])
    gift_happiness_collect = SharedArray{Float64}(N_GIFT_TYPE,init=0)
    # for gift_id in 0:size(gift_happiness_collect)[1]-1
    #     gift_happiness_collect[gift_id+1] = 0
    # end
    # total_gift_happiness = zeros(N_GIFT_TYPE)

    @parallel for (child_id, gift_id) in gift_assignment

        # check if child_id and gift_id exist
        # assert child_id < N_CHILDREN
        # assert gift_id < N_GIFT_TYPE
        # assert child_id >= 0
        # assert gift_id >= 0
        #println(size(gift_pref[child_id+1,:]))
        gift_rank = find(x -> x == gift_id, gift_pref[child_id+1,:])
        if length(gift_rank) == 0
           child_happiness = -1
        else
           child_happiness = (N_GIFT_PREF - gift_rank[1]) * RATIO_CHILD_HAPPINESS
        end

        child_rank = find(x -> x == child_id, child_pref[gift_id + 1,:])
        if length(child_rank) == 0
           gift_happiness = -1
        else
           gift_happiness = (N_CHILD_PREF - child_rank[1]) * RATIO_GIFT_HAPPINESS
        end

        child_happiness_collect[child_id+1] = child_happiness
        @show child_happiness_collect[child_id+1]
        println(child_id+1, child_happiness, child_happiness_collect[child_id+1])
        gift_happiness_collect[gift_id+1] += gift_happiness
        @show gift_happiness_collect[gift_id+1]
    end

    print(child_happiness_collect)
    total_child_happiness = sum(child_happiness_collect)
    total_gift_happiness = gift_happiness_collect
    # for gift_id in 0:size(gift_happiness_collect)[1]-1
    #     for child_id in 0:size(gift_happiness_collect)[2]-1
    #         total_gift_happiness[gift_id+1] += gift_happiness_collect[gift_id+1, child_id+1]
    #     end
    # end

    # total_gift_happiness[gift_id+1] += gift_happiness
    println("parallel happines compute time: ", toc())
    println("parallel happiness: ", total_child_happiness)
    println("paralel gift happinaess: ", total_gift_happiness)


    println("normalized child happiness=",Float16(total_child_happiness)/(Float16(N_CHILDREN)*Float16(max_child_happiness)) ,
        ", normalized gift happiness",mean(total_gift_happiness) / Float16(max_gift_happiness*N_GIFT_QUANTITY))

    # to avoid float rounding error
    # find common denominator
    # NOTE: I used this code to experiment different parameters, so it was necessary to get the multiplier
    # Note: You should hard-code the multipler to speed up, now that the parameters are finalized
    denominator1 = N_CHILDREN*max_child_happiness
    denominator2 = N_GIFT_QUANTITY*max_gift_happiness*N_GIFT_TYPE
    common_denom = lcm(denominator1, denominator2)
    multiplier = common_denom / denominator1

    # # usually denom1 > demon2
    return ((total_child_happiness*multiplier) ^ 3 + (sum(total_gift_happiness) ^3)) / (common_denom ^ 3)

end
