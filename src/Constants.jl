module Constants


export N_CHILDREN, N_GIFT_TYPE, N_GIFT_QUANTITY, N_GIFT_PREF, N_CHILD_PREF
export TWINS, TRIPLETS, TRIPLETS, RATIO_GIFT_HAPPINESS, RATIO_CHILD_HAPPINESS
export TRIPLETS_INDEX, TWINS_INDEX, TWINS_RANGE, SINGLE_INDEX
export WISHLIST_FILE, GOODKIDS_FILE, PROCESSED_DATA_FILE
export MAX_GIFT_HAPPINESS, MAX_CHILD_HAPPINESS


const N_CHILDREN = 1000000 # n children to give
const N_GIFT_TYPE = 1000 # n types of gifts available
const N_GIFT_QUANTITY = 1000 # each type of gifts are limited to this quantity
const N_GIFT_PREF = 100 # number of gifts a child ranks
const N_CHILD_PREF = 1000 # number of children a gift ranks
const TWINS = ceil(0.04 * N_CHILDREN / 2.) * 2    # 4% of all population, rounded to the closest number
const TRIPLETS = ceil(0.005 * N_CHILDREN / 3.) * 3    # 0.5% of all population, rounded to the closest number
const RATIO_GIFT_HAPPINESS = 2
const RATIO_CHILD_HAPPINESS = 2

const MAX_CHILD_HAPPINESS = N_GIFT_PREF * RATIO_CHILD_HAPPINESS
const MAX_GIFT_HAPPINESS = N_CHILD_PREF * RATIO_GIFT_HAPPINESS



# index of first members of twins and triplets
const TRIPLETS_INDEX = colon(0,3, TRIPLETS-1)
const TWINS_INDEX = colon(TRIPLETS,2,TRIPLETS+TWINS-2)
const TWINS_RANGE = [TRIPLETS, TRIPLETS+TWINS-1]
const SINGLE_INDEX = colon(TRIPLETS + TWINS ,1,N_CHILDREN-1)

# [i for i in TRIPLETS_INDEX]
# [i for i in TWINS_INDEX]

const WISHLIST_FILE = "data/input/child_wishlist_v2.csv"
const GOODKIDS_FILE = "data/input/gift_goodkids_v2.csv"

const PROCESSED_DATA_FILE = "data/working/data.jld"
const PROCESSED_DATA_PATH = joinpath(pwd(),PROCESSED_DATA_FILE)



end
