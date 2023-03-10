"""
element specific data
"""

import re

# Unit conversion
bohr_to_angs = 0.529177249

atom_names = ["h", "he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg", "al", "si", "p", "s", "cl", "ar", "k", "ca", "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn", "ga", "ge", "as", "se", "br", "kr", "rb", "sr", "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd", "in", "sn", "sb", "te", "i", "xe","cs", "ba", "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb", "dy", "ho", "er", "tm", "yb", "lu", "hf", "ta", "w", "re", "os", "ir", "pt", "au", "hg", "tl", "pb", "bi", "po", "at", "rn",
"fr", "ra", "ac", "th", "pa", "u", "np", "pu", "am", "cm", "bk", "cf", "es", "fm", "md", "no", "lr",
"rf", "db", "sg", "bh", "hs", "mt", "ds", "rg", "uub", "uut", "uuq", "uup", "uuh"]

def atomic_number(atom):
    # remove integers meant to label inequivalent atoms of the same element, C12 -> C
    elem = re.sub('[0-9]', '', atom.lower())
    atno = atom_names.index(elem) + 1
    
    return atno

def element_name(Z):
    """translate atomic number Z into element name"""
    return atom_names[Z-1]

# masses in atomic units
# I believe these masses are averaged over isotopes weighted with their abundances
atom_masses = {"h" : 1.837362128065067E+03, 	 "he" : 7.296296732461748E+03, 	 
 "li" : 1.265266834424631E+04, 	 "be" : 1.642820197435333E+04, 	 "b" : 1.970724642985837E+04, 	 "c" : 2.189416563639810E+04, 	 "n" : 2.553265087125124E+04, 	 "o" : 2.916512057440347E+04, 	 "f" : 3.463378575704848E+04, 	 "ne" : 3.678534092874044E+04, 	 "na" : 4.190778360619254E+04, 	 "mg" : 4.430530242139558E+04, 	 "al" : 4.918433357200404E+04, 	 "si" : 5.119673199572539E+04, 	 "p" : 5.646171127497759E+04, 	 "s" : 5.845091636050397E+04, 	 "cl" : 6.462686224010440E+04, 	 "ar" : 7.282074557210083E+04, 	 "k" : 7.127183730353635E+04, 	 "ca" : 7.305772106334878E+04, 	 "sc" : 8.194961023615085E+04, 	 "ti" : 8.725619876588941E+04, 	 "v" : 9.286066913390342E+04, 	 "cr" : 9.478308723444256E+04, 	 "mn" : 1.001459246313614E+05, 	 "fe" : 1.017992023749367E+05, 	 "co" : 1.074286371995095E+05, 	 "ni" : 1.069915176770187E+05, 	 "cu" : 1.158372658987864E+05, 	 "zn" : 1.191804432137767E+05, 	 "ga" : 1.270972475098524E+05, 	 "ge" : 1.324146129557776E+05, 	 "as" : 1.365737151160185E+05, 	 "se" : 1.439352676072164E+05, 	 "br" : 1.456560742513554E+05, 	 "kr" : 1.527544016584286E+05, 	 "rb" : 1.557982606990888E+05, 	 "sr" : 1.597214811011183E+05, 	 "y" : 1.620654421428197E+05, 	 "zr" : 1.662911708738692E+05, 	 "nb" : 1.693579618505286E+05, 	 "mo" : 1.749243703088714E+05, 	 "tc" : 1.786430626330700E+05, 	 "ru" : 1.842393300033101E+05, 	 "rh" : 1.875852416508917E+05, 	 "pd" : 1.939917829123603E+05, 	 "ag" : 1.966316898848625E+05, 	 "cd" : 2.049127072821024E+05, 	 "in" : 2.093003996469779E+05, 	 "sn" : 2.163950812772627E+05, 	 "sb" : 2.219548908796184E+05, 	 "te" : 2.326005591018340E+05, 	 "i" : 2.313326855370057E+05, 	 "xe" : 2.393324859416700E+05, 	 "cs" : 2.422718057964099E+05, 	 "ba" : 2.503317945123633E+05, 	 "la" : 2.532091691559799E+05, 	 "ce" : 2.554158302438290E+05, 	 "pr" : 2.568589198411092E+05, 	 "nd" : 2.629370677583600E+05, 	 "pm" : 2.643188171611750E+05, 	 "sm" : 2.740894989541674E+05, 	 "eu" : 2.770134119384883E+05, 	 "gd" : 2.866491999903088E+05, 	 "tb" : 2.897031760615568E+05, 	 "dy" : 2.962195463487770E+05, 	 "ho" : 3.006495661821661E+05, 	 "er" : 3.048944899280067E+05, 	 "tm" : 3.079482107948796E+05, 	 "yb" : 3.154581281724826E+05, 	 "lu" : 3.189449490929371E+05, 	 "hf" : 3.253673494834354E+05, 	 "ta" : 3.298477904098085E+05, 	 "w" : 3.351198023924856E+05, 	 "re" : 3.394345792215925E+05, 	 "os" : 3.467680592315195E+05, 	 "ir" : 3.503901384708247E+05, 	 "pt" : 3.501476943143941E+05, 	 "au" : 3.590480726784480E+05, 	 "hg" : 3.656531829955869E+05, 	 "tl" : 3.725679455413626E+05, 	 "pb" : 3.777024752813480E+05, 	 "bi" : 3.809479476012968E+05, 	 "po" : 3.828065627851500E+05, 	 "at" : 3.828065627851500E+05, 	 "rn" : 4.010354467273000E+05, 	 "fr" : 4.065041119099450E+05, 	 "ra" : 4.119727770925900E+05, 	 "ac" : 4.137956654868050E+05, 	 "th" : 4.229794865901638E+05, 	 "pa" : 4.211526242992494E+05, 	 "u" : 4.339001375266468E+05, 	 "np" : 4.320245494289550E+05, 	 "pu" : 4.447847681884600E+05, 	 "am" : 4.429618797942450E+05, 	 "cm" : 4.502534333711050E+05, 	 "bk" : 4.502534333711050E+05, 	 "cf" : 4.575449869479650E+05, 	 "es" : 4.593678753421800E+05, 	 "fm" : 4.684823173132550E+05, 	 "md" : 4.703052057074700E+05, 	 "no" : 4.721280941016850E+05, 	 "lr" : 4.775967592843300E+05, 	 "rf" : 4.757738708901150E+05, 	 "db" : 4.775967592843300E+05, 	 "sg" : 4.848883128611900E+05, 	 "bh" : 4.812425360727600E+05, 	 "hs" : 5.049400851975550E+05, 	 "mt" : 4.885340896496200E+05, 	 "ds" : 4.940027548322650E+05, 	 "rg" : 4.958256432264800E+05, 	 "uub" : 5.195231923512750E+05, 	 "uut" : 5.177003039570600E+05, 	 "uuq" : 5.268147459281350E+05, 	 "uup" : 5.249918575339200E+05, 	 "uuh" : 5.322834111107800E+05, 	 "uuo" : 5.359291878992100E+05 }


def atomlist2masses(atomlist):
    """
    assign masses to atoms. The mass is repeated for each coordinate
    """
    masses = []
    for (Zi, posi) in atomlist:
        mi = atom_masses[atom_names[Zi-1]]
        masses += [mi, mi, mi]
    return masses

# atomic radii taken from Cordero el.al.,"Covalent radii revisited", Dalton Trans., 2008, 2832-2838
# in bohr, in the paper the values are given in Angstrom with 2 decimals
covalent_radii = {k : v/bohr_to_angs for (k,v) in
                  {
    "h" : 0.31, "he": 0.28, "li": 1.28, "be": 0.96, "b" : 0.84,
    "c" : 0.76, "n" : 0.71, "o" : 0.66, "f" : 0.57, "ne": 0.58,
    "na": 1.66, "mg": 1.41, "al": 1.21, "si": 1.11, "p" : 1.07,
    "s" : 1.05, "cl": 1.02, "ar": 1.06, "k" : 2.03, "ca": 1.76,
    "sc": 1.70, "ti": 1.60, "v" : 1.53, "cr": 1.39, "mn": 1.39,
    "fe": 1.32, "co": 1.26, "ni": 1.24, "cu": 1.32, "zn": 1.22,
    "ga": 1.22, "ge": 1.20, "as": 1.19, "se": 1.20, "br": 1.20,
    "kr": 1.16, "rb": 2.20, "sr": 1.95, "y" : 1.90, "zr": 1.75,
    "nb": 1.64, "mo": 1.54, "tc": 1.47, "ru": 1.46, "rh": 1.42,
    "pd": 1.39, "ag": 1.45, "cd": 1.44, "in": 1.42, "sn": 1.39,
    "sb": 1.39, "te": 1.38, "i" : 1.39, "xe": 1.40, "cs": 2.44,
    "ba": 2.15, "la": 2.07, "ce": 2.04, "pr": 2.03, "nd": 2.01,
    "pm": 1.99, "sm": 1.98, "eu": 1.98, "gd": 1.96, "tb": 1.94,
    "dy": 1.92, "ho": 1.92, "er": 1.89, "tm": 1.90, "yb": 1.87,
    "lu": 1.87, "hf": 1.75, "ta": 1.70, "w" : 1.62, "re": 1.51,
    "os": 1.44, "ir": 1.41, "pt": 1.36, "au": 1.36, "hg": 1.32,
    "tl": 1.45, "pb": 1.46, "bi": 1.48, "po": 1.40, "at": 1.50,
    "rn": 1.50, "fr": 2.60, "ra": 2.21, "ac": 2.15, "th": 2.06,
    "pa": 2.00, "u" : 1.96, "np": 1.90, "pu": 1.87, "am": 1.80,
    "cm": 1.69
                  }.items()}

# occupation numbers of valence orbitals
# which are used to assign the correct occupation to orbitals loaded from hotbit .elm files
valence_occ_numbers = {"h": {"1s": 1}, "he": {"1s": 2}, \
"li": {"2s": 1}, "be": {"2s": 2}, "b": {"2s": 2, "2p": 1}, "c": {"2s": 2, "2p": 2}, "n": {"2s": 2, "2p": 3}, "o": {"2s": 2, "2p": 4}, "f": {"2s": 2, "2p": 5}, "ne": {"2s": 2, "2p": 6}, \
"na": {"3s": 1}, "mg": {"3s": 2}, "al": {"3s": 2, "3p": 1}, "si": {"3s": 2, "3p": 2}, "p": {"3s": 2, "3p": 3}, "s": {"3s": 2, "3p": 4}, "cl": {"3s": 2, "3p": 5}, "ar": {"3s": 2, "3p": 6}, \
    "sc": {"3d": 1, "4s": 2}, "ti": {"3d": 2, "4s": 2}, "zn": {"3d": 10, "4s": 2}, "br": {"4s": 2, "4p": 5}, \
"ru": {"4d": 7, "5s": 1}, \
"i": {"5s": 2, "5p": 5}, \
"au": {"4f": 14, "5d": 10, "6s": 1}\
}

# number of valence electrons
valence_electrons = dict([(at, sum(valence_occ_numbers[at].values())) for at in valence_occ_numbers.keys()])
