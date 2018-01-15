
module Utils

using CSV

include("Constants.jl")
using Constants

export make_dummy_output, check_feas, avg_normalized_happiness, load_data



function load_data()
    gift_pref_df = CSV.read(joinpath(pwd(),WISHLIST_FILE), header=false, nullable=false)
    child_pref_df = CSV.read(joinpath(pwd(),GOODKIDS_FILE), header=false, nullable=false)

    # head(gift_pref_df)
    # head(child_pref_df)



    gift_pref = convert(Array, gift_pref_df[:, 2:end])
    child_pref = convert(Array, child_pref_df[:, 2:end])
    # size(child_pref)
    # size(gift_pref)


    clear!(:gift_pref_df)
    clear!(:child_pref_df)

    return gift_pref, child_pref
end


function make_dummy_output()
    output = Array{Int}((N_CHILDREN,2))

    for i in 1:size(output)[1]
        try
            output[i,1] = i - 1
            output[i,2] = (i-1) % N_GIFT_TYPE
        catch
            @show i
        end
    end
    return output
end

function check_feas(output)
    # check if triplets have the same gift
    for t1 in TRIPLETS_INDEX #colon(0,3, TRIPLETS-1)
        # @show convert(Int,t1)
        index = convert(Int,t1) +1
        triplet1 = output[index, 2]
        triplet2 = output[index+1,2]
        triplet3 = output[index+2,2]
        # println("$index:$triplet1, $(index+1):$triplet2, $(index+2):$triplet3")
        #throw(AssertionError("$(triplet1[1]) == $(triplet2[1]) and $(triplet2[1]) == $(triplet3[1])"))
        if (triplet1[1] != triplet2[1]) | (triplet2[1] != triplet3[1])
            warn("infeasible triplet assignemnt")
            return false

        end
    end

    # check if twins have the same gift
    for t1 in TWINS_INDEX #colon(TRIPLETS-1,2,TRIPLETS+TWINS-2)

        index = convert(Int,t1) +1
        twin1 = output[index, 2]
        twin2 = output[index+1,2]
        #println("$index:$twin1, $(index+1):$twin2")
        if (twin1[1] != twin2[1])
            warn("infeasible twin assignemnt")
            return false

        end
    end

    gift_counts = fill!(Array{Int64,1}(N_GIFT_TYPE), 0) #Dict(v=>0 for (k,v) in gift_assignment )
    tic()
    for kid in 0:N_CHILDREN-1
        gift = output[kid+1,2]
        gift_counts[gift+1] += 1
        if gift_counts[gift+1] > N_GIFT_QUANTITY
            warn("infeasible gift quantity assignement")
            # throw(AssertionError("quantity for $gift==$(gift_counts[gift]) >=$N_GIFT_QUANTITY"))
            return false
        end
    end
    return true
end

function avg_normalized_happiness(output, child_pref, gift_pref)
    # zip(output[:,1], output[:,2])
    gift_assignment = Dict(zip(output[:,1] , output[:,2]))
    gift_counts = Dict(v=>0 for (k,v) in gift_assignment )
    tic()
    for (kid, gift) in gift_assignment
        gift_counts[gift] += 1
        if gift_counts[gift] > N_GIFT_QUANTITY
            throw(AssertionError("quantity for $gift==$(gift_counts[gift]) >=$N_GIFT_QUANTITY"))
        end
    end
    println("couting gifts ",toc())
    println("checking feasibility: ", check_feas(output))

    max_child_happiness = N_GIFT_PREF * RATIO_CHILD_HAPPINESS
    max_gift_happiness = N_CHILD_PREF * RATIO_GIFT_HAPPINESS

    total_child_happiness = 0
    total_gift_happiness = zeros(N_GIFT_TYPE)

    tic()
    for (child_id, gift_id) in gift_assignment

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

        total_child_happiness += child_happiness
        total_gift_happiness[gift_id+1] += gift_happiness

    end
    println("total_child_happiness ", total_child_happiness)
    println("total_gift_happiness ", sum(total_gift_happiness))

    println("normalized child happiness=",total_child_happiness/(N_CHILDREN *max_child_happiness) ,
        ", normalized gift happiness",mean(total_gift_happiness) / (max_gift_happiness*N_GIFT_QUANTITY))

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


end
