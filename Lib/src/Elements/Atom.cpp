/* Copyright 2017 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @file Atom.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Atom.h"

using namespace std;

namespace TBTK{

string Atom::symbols[Atom::totalNumAtoms+1] = {
	"None",
	"H",
	"He",
	"Li",
	"Be",
	"B",
	"C",
	"N",
	"O",
	"F",
	"Ne",	//10
	"Na",
	"Mg",
	"Al",
	"Si",
	"P",
	"S",
	"Cl",
	"Ar",
	"K",
	"Ca",	//20
	"Sc",
	"Ti",
	"V",
	"Cr",
	"Mn",
	"Fe",
	"Co",
	"Ni",
	"Cu",
	"Zn",	//30
	"Ga",
	"Ge",
	"As",
	"Se",
	"Br",
	"Kr",
	"Rb",
	"Sr",
	"Y",
	"Zr",	//40
	"Nb",
	"Mo",
	"Tc",
	"Ru",
	"Rh",
	"Pd",
	"Ag",
	"Cd",
	"In",
	"Sn",	//50
	"Sb",
	"Te",
	"I",
	"Xe",
	"Cs",
	"Ba",
	"La",
	"Ce",
	"Pr",
	"Nd",	//60
	"Pm",
	"Sm",
	"Eu",
	"Gd",
	"Tb",
	"Dy",
	"Ho",
	"Er",
	"Tm",
	"Yb",	//70
	"Lu",
	"Hf",
	"Ta",
	"W",
	"Re",
	"Os",
	"Ir",
	"Pt",
	"Au",
	"Hg",	//80
	"Tl",
	"Pb",
	"Bi",
	"Po",
	"At",
	"Rn",
	"Fr",
	"Ra",
	"Ac",
	"Th",	//90
	"Pa",
	"U",
	"Np",
	"Pu",
	"Am",
	"Cm",
	"Bk",
	"Cf",
	"Es",
	"Fm",	//100
	"Md",
	"No",
	"Lr",
	"Rf",
	"Db",
	"Sg",
	"Bh",
	"Hs",
	"Mt",
	"Ds",	//110
	"Rg",
	"Cn",
	"Nh",
	"Fl",
	"Mc",
	"Lv",
	"Ts",
	"Og"
};

string Atom::names[Atom::totalNumAtoms+1] = {
	"None",
	"Hydrogen",
	"Helium",
	"Lithium",
	"Beryllium",
	"Boron",
	"Carbon",
	"Nitrogen",
	"Oxygen",
	"Fluorine",
	"Neon",		//10
	"Sodium",
	"Magnesium",
	"Aluminium",
	"Silicon",
	"Phosphorus",
	"Sulfur",
	"Chlorine",
	"Argon",
	"Potassium",
	"Calcium",	//20
	"Scandium",
	"Titanium",
	"Vanadium",
	"Chromium",
	"Manganese",
	"Iron",
	"Cobalt",
	"Nickel",
	"Copper",
	"Zinc",		//30
	"Gallium",
	"Germanium",
	"Arsenic",
	"Selenium",
	"Bromine",
	"Krypton",
	"Rubidium",
	"Strontium",
	"Yttrium",
	"Zirconium",	//40
	"Niobium",
	"Molybdenum",
	"Technetium",
	"Ruthenium",
	"Rhodium",
	"Palladium",
	"Silver",
	"Cadmium",
	"Indium",
	"Tin",		//50
	"Antimony",
	"Tellurium",
	"Iodine",
	"Xenon",
	"Caesium",
	"Barium",
	"Lanthanum",
	"Cerium",
	"Praseodymium",
	"Neodymium",	//60
	"Promethium",
	"Samarium",
	"Europium",
	"Gadolinium",
	"Terbium",
	"Dysprosium",
	"Holmium",
	"Erbium",
	"Thulium",
	"Ytterbium",	//70
	"Lutetium",
	"Hafnium",
	"Tantalum",
	"Tungsten",
	"Rhenium",
	"Osmium",
	"Iridium",
	"Platinum",
	"Gold",
	"Mercury",	//80
	"Thallium",
	"Lead",
	"Bismuth",
	"Polonium",
	"Astatine",
	"Radon",
	"Francium",
	"Radium",
	"Actinium",
	"Thorium",	//90
	"Protactinium",
	"Uranium",
	"Neptunium",
	"Plutonium",
	"Americium",
	"Curium",
	"Berkelium",
	"Californium",
	"Einsteinium",
	"Fermium",	//100
	"Mendelevium",
	"Nobelium",
	"Lawrencium",
	"Rutherfordium",
	"Dubnium",
	"Seaborgium",
	"Bohrium",
	"Hassium",
	"Meitnerium",
	"Darmstadtium",	//110
	"Roentgenium",
	"Copernicium",
	"Nihonium",
	"Flerovium",
	"Moscovium",
	"Livermorium",
	"Tennessine",
	"Oganesson"
};

double Atom::standardWeights[totalNumAtoms+1] = {
	0,
	1.008,
	4.0026,
	6.94,
	9.0122,
	10.81,
	12.011,
	14.007,
	15.999,
	18.998,
	20.180,	//10
	22.990,
	24.305,
	26.982,
	28.085,
	30.974,
	32.06,
	35.45,
	39.948,
	39.098,
	40.078,	//20
	44.956,
	47.867,
	50.942,
	51.996,
	54.938,
	55.845,
	58.933,
	58.693,
	63.546,
	65.38,	//30
	69.723,
	72.630,
	74.922,
	78.971,
	79.904,
	83.798,
	85.468,
	87.62,
	88.906,
	91.224,	//40
	92.906,
	95.95,
	98,
	101.07,
	102.91,
	106.42,
	107.87,
	112.41,
	114.82,
	118.71,	//50
	121.76,
	127.60,
	126.90,
	131.29,
	132.91,
	137.33,
	138.91,
	140.12,
	140.91,
	144.24,	//60
	145,
	150.36,
	151.96,
	157.25,
	158.93,
	162.50,
	164.93,
	167.26,
	168.93,
	173.05,	//70
	174.97,
	178.49,
	180.95,
	183.84,
	186.21,
	190.23,
	192.22,
	195.08,
	196.97,
	200.59,	//80
	204.38,
	207.2,
	208.98,
	209,
	210,
	222,
	223,
	226,
	227,
	232.04,	//90
	231.04,
	238.03,
	237,
	244,
	243,
	247,
	247,
	251,
	252,
	257,	//100
	258,
	259,
	266,
	267,
	268,
	269,
	270,
	277,
	278,
	281,	//110
	282,
	285,
	286,
	289,
	290,
	293,
	294,
	294
};

Atom Atom::atoms[Atom::totalNumAtoms+1] = {
	Atom(0),
	Atom(1),
	Atom(2),
	Atom(3),
	Atom(4),
	Atom(5),
	Atom(6),
	Atom(7),
	Atom(8),
	Atom(9),
	Atom(10),
	Atom(11),
	Atom(12),
	Atom(13),
	Atom(14),
	Atom(15),
	Atom(16),
	Atom(17),
	Atom(18),
	Atom(19),
	Atom(20),
	Atom(21),
	Atom(22),
	Atom(23),
	Atom(24),
	Atom(25),
	Atom(26),
	Atom(27),
	Atom(28),
	Atom(29),
	Atom(30),
	Atom(31),
	Atom(32),
	Atom(33),
	Atom(34),
	Atom(35),
	Atom(36),
	Atom(37),
	Atom(38),
	Atom(39),
	Atom(40),
	Atom(41),
	Atom(42),
	Atom(43),
	Atom(44),
	Atom(45),
	Atom(46),
	Atom(47),
	Atom(48),
	Atom(49),
	Atom(50),
	Atom(51),
	Atom(52),
	Atom(53),
	Atom(54),
	Atom(55),
	Atom(56),
	Atom(57),
	Atom(58),
	Atom(59),
	Atom(60),
	Atom(61),
	Atom(62),
	Atom(63),
	Atom(64),
	Atom(65),
	Atom(66),
	Atom(67),
	Atom(68),
	Atom(69),
	Atom(70),
	Atom(71),
	Atom(72),
	Atom(73),
	Atom(74),
	Atom(75),
	Atom(76),
	Atom(77),
	Atom(78),
	Atom(79),
	Atom(80),
	Atom(81),
	Atom(82),
	Atom(83),
	Atom(84),
	Atom(85),
	Atom(86),
	Atom(87),
	Atom(88),
	Atom(89),
	Atom(90),
	Atom(91),
	Atom(92),
	Atom(93),
	Atom(94),
	Atom(95),
	Atom(96),
	Atom(97),
	Atom(98),
	Atom(99),
	Atom(100),
	Atom(101),
	Atom(102),
	Atom(103),
	Atom(104),
	Atom(105),
	Atom(106),
	Atom(107),
	Atom(108),
	Atom(109),
	Atom(110),
	Atom(111),
	Atom(112),
	Atom(113),
	Atom(114),
	Atom(115),
	Atom(116),
	Atom(117),
	Atom(118)
};

const Atom &Atom::Hydrogen	= atoms[1];
const Atom &Atom::Helium	= atoms[2];
const Atom &Atom::Lithium	= atoms[3];
const Atom &Atom::Beryllium	= atoms[4];
const Atom &Atom::Boron		= atoms[5];
const Atom &Atom::Carbon	= atoms[6];
const Atom &Atom::Nitrogen	= atoms[7];
const Atom &Atom::Oxygen	= atoms[8];
const Atom &Atom::Fluorine	= atoms[9];
const Atom &Atom::Neon		= atoms[10];
const Atom &Atom::Sodium	= atoms[11];
const Atom &Atom::Magnesium	= atoms[12];
const Atom &Atom::Aluminium	= atoms[13];
const Atom &Atom::Silicon	= atoms[14];
const Atom &Atom::Phosphorus	= atoms[15];
const Atom &Atom::Sulfur	= atoms[16];
const Atom &Atom::Chlorine	= atoms[17];
const Atom &Atom::Argon		= atoms[18];
const Atom &Atom::Potassium	= atoms[19];
const Atom &Atom::Calcium	= atoms[20];
const Atom &Atom::Scandium	= atoms[21];
const Atom &Atom::Titanium	= atoms[22];
const Atom &Atom::Vanadium	= atoms[23];
const Atom &Atom::Chromium	= atoms[24];
const Atom &Atom::Manganese	= atoms[25];
const Atom &Atom::Iron		= atoms[26];
const Atom &Atom::Cobalt	= atoms[27];
const Atom &Atom::Nickel	= atoms[28];
const Atom &Atom::Copper	= atoms[29];
const Atom &Atom::Zinc		= atoms[30];
const Atom &Atom::Gallium	= atoms[31];
const Atom &Atom::Germanium	= atoms[32];
const Atom &Atom::Arsenic	= atoms[33];
const Atom &Atom::Selenium	= atoms[34];
const Atom &Atom::Bromine	= atoms[35];
const Atom &Atom::Krypton	= atoms[36];
const Atom &Atom::Rubidium	= atoms[37];
const Atom &Atom::Strontium	= atoms[38];
const Atom &Atom::Yttrium	= atoms[39];
const Atom &Atom::Zirconium	= atoms[40];
const Atom &Atom::Niobium	= atoms[41];
const Atom &Atom::Molybdenum	= atoms[42];
const Atom &Atom::Technetium	= atoms[43];
const Atom &Atom::Ruthenium	= atoms[44];
const Atom &Atom::Rhodium	= atoms[45];
const Atom &Atom::Palladium	= atoms[46];
const Atom &Atom::Silver	= atoms[47];
const Atom &Atom::Cadmium	= atoms[48];
const Atom &Atom::Indium	= atoms[49];
const Atom &Atom::Tin		= atoms[50];
const Atom &Atom::Antimony	= atoms[51];
const Atom &Atom::Tellurium	= atoms[52];
const Atom &Atom::Iodine	= atoms[53];
const Atom &Atom::Xenon		= atoms[54];
const Atom &Atom::Caesium	= atoms[55];
const Atom &Atom::Barium	= atoms[56];
const Atom &Atom::Lanthanum	= atoms[57];
const Atom &Atom::Cerium	= atoms[58];
const Atom &Atom::Praseodymium	= atoms[59];
const Atom &Atom::Neodymium	= atoms[60];
const Atom &Atom::Promethium	= atoms[61];
const Atom &Atom::Samarium	= atoms[62];
const Atom &Atom::Europium	= atoms[63];
const Atom &Atom::Gadolinium	= atoms[64];
const Atom &Atom::Terbium	= atoms[65];
const Atom &Atom::Dysprosium	= atoms[66];
const Atom &Atom::Holmium	= atoms[67];
const Atom &Atom::Erbium	= atoms[68];
const Atom &Atom::Thulium	= atoms[69];
const Atom &Atom::Ytterbium	= atoms[70];
const Atom &Atom::Lutetium	= atoms[71];
const Atom &Atom::Hafnium	= atoms[72];
const Atom &Atom::Tantalum	= atoms[73];
const Atom &Atom::Tungsten	= atoms[74];
const Atom &Atom::Rhenium	= atoms[75];
const Atom &Atom::Osmium	= atoms[76];
const Atom &Atom::Iridium	= atoms[77];
const Atom &Atom::Platinum	= atoms[78];
const Atom &Atom::Gold		= atoms[79];
const Atom &Atom::Mercury	= atoms[80];
const Atom &Atom::Thallium	= atoms[81];
const Atom &Atom::Lead		= atoms[82];
const Atom &Atom::Bismuth	= atoms[83];
const Atom &Atom::Polonium	= atoms[84];
const Atom &Atom::Astatine	= atoms[85];
const Atom &Atom::Radon		= atoms[86];
const Atom &Atom::Francium	= atoms[87];
const Atom &Atom::Radium	= atoms[88];
const Atom &Atom::Actinium	= atoms[89];
const Atom &Atom::Thorium	= atoms[90];
const Atom &Atom::Protactinium	= atoms[91];
const Atom &Atom::Uranium	= atoms[92];
const Atom &Atom::Neptunium	= atoms[93];
const Atom &Atom::Plutonium	= atoms[94];
const Atom &Atom::Americium	= atoms[95];
const Atom &Atom::Curium	= atoms[96];
const Atom &Atom::Berkelium	= atoms[97];
const Atom &Atom::Californium	= atoms[98];
const Atom &Atom::Einsteinium	= atoms[99];
const Atom &Atom::Fermium	= atoms[100];
const Atom &Atom::Mendelevium	= atoms[101];
const Atom &Atom::Nobelium	= atoms[102];
const Atom &Atom::Lawrencium	= atoms[103];
const Atom &Atom::Rutherfordium	= atoms[104];
const Atom &Atom::Dubnium	= atoms[105];
const Atom &Atom::Seaborgium	= atoms[106];
const Atom &Atom::Bohrium	= atoms[107];
const Atom &Atom::Hassium	= atoms[108];
const Atom &Atom::Meitnerium	= atoms[109];
const Atom &Atom::Darmstadtium	= atoms[110];
const Atom &Atom::Roentgenium	= atoms[111];
const Atom &Atom::Copernicium	= atoms[112];
const Atom &Atom::Nihonium	= atoms[113];
const Atom &Atom::Flerovium	= atoms[114];
const Atom &Atom::Moscovium	= atoms[115];
const Atom &Atom::Livermorium	= atoms[116];
const Atom &Atom::Tennessine	= atoms[117];
const Atom &Atom::Oganesson	= atoms[118];

const Atom &Atom::H	= atoms[1];
const Atom &Atom::He	= atoms[2];
const Atom &Atom::Li	= atoms[3];
const Atom &Atom::Be	= atoms[4];
const Atom &Atom::B	= atoms[5];
const Atom &Atom::C	= atoms[6];
const Atom &Atom::N	= atoms[7];
const Atom &Atom::O	= atoms[8];
const Atom &Atom::F	= atoms[9];
const Atom &Atom::Ne	= atoms[10];
const Atom &Atom::Na	= atoms[11];
const Atom &Atom::Mg	= atoms[12];
const Atom &Atom::Al	= atoms[13];
const Atom &Atom::Si	= atoms[14];
const Atom &Atom::P	= atoms[15];
const Atom &Atom::S	= atoms[16];
const Atom &Atom::Cl	= atoms[17];
const Atom &Atom::Ar	= atoms[18];
const Atom &Atom::K	= atoms[19];
const Atom &Atom::Ca	= atoms[20];
const Atom &Atom::Sc	= atoms[21];
const Atom &Atom::Ti	= atoms[22];
const Atom &Atom::V	= atoms[23];
const Atom &Atom::Cr	= atoms[24];
const Atom &Atom::Mn	= atoms[25];
const Atom &Atom::Fe	= atoms[26];
const Atom &Atom::Co	= atoms[27];
const Atom &Atom::Ni	= atoms[28];
const Atom &Atom::Cu	= atoms[29];
const Atom &Atom::Zn	= atoms[30];
const Atom &Atom::Ga	= atoms[31];
const Atom &Atom::Ge	= atoms[32];
const Atom &Atom::As	= atoms[33];
const Atom &Atom::Se	= atoms[34];
const Atom &Atom::Br	= atoms[35];
const Atom &Atom::Kr	= atoms[36];
const Atom &Atom::Rb	= atoms[37];
const Atom &Atom::Sr	= atoms[38];
const Atom &Atom::Y	= atoms[39];
const Atom &Atom::Zr	= atoms[40];
const Atom &Atom::Nb	= atoms[41];
const Atom &Atom::Mo	= atoms[42];
const Atom &Atom::Tc	= atoms[43];
const Atom &Atom::Ru	= atoms[44];
const Atom &Atom::Rh	= atoms[45];
const Atom &Atom::Pd	= atoms[46];
const Atom &Atom::Ag	= atoms[47];
const Atom &Atom::Cd	= atoms[48];
const Atom &Atom::In	= atoms[49];
const Atom &Atom::Sn	= atoms[50];
const Atom &Atom::Sb	= atoms[51];
const Atom &Atom::Te	= atoms[52];
const Atom &Atom::I	= atoms[53];
const Atom &Atom::Xe	= atoms[54];
const Atom &Atom::Cs	= atoms[55];
const Atom &Atom::Ba	= atoms[56];
const Atom &Atom::La	= atoms[57];
const Atom &Atom::Ce	= atoms[58];
const Atom &Atom::Pr	= atoms[59];
const Atom &Atom::Nd	= atoms[60];
const Atom &Atom::Pm	= atoms[61];
const Atom &Atom::Sm	= atoms[62];
const Atom &Atom::Eu	= atoms[63];
const Atom &Atom::Gd	= atoms[64];
const Atom &Atom::Tb	= atoms[65];
const Atom &Atom::Dy	= atoms[66];
const Atom &Atom::Ho	= atoms[67];
const Atom &Atom::Er	= atoms[68];
const Atom &Atom::Tm	= atoms[69];
const Atom &Atom::Yb	= atoms[70];
const Atom &Atom::Lu	= atoms[71];
const Atom &Atom::Hf	= atoms[72];
const Atom &Atom::Ta	= atoms[73];
const Atom &Atom::W	= atoms[74];
const Atom &Atom::Re	= atoms[75];
const Atom &Atom::Os	= atoms[76];
const Atom &Atom::Ir	= atoms[77];
const Atom &Atom::Pt	= atoms[78];
const Atom &Atom::Au	= atoms[79];
const Atom &Atom::Hg	= atoms[80];
const Atom &Atom::Tl	= atoms[81];
const Atom &Atom::Pb	= atoms[82];
const Atom &Atom::Bi	= atoms[83];
const Atom &Atom::Po	= atoms[84];
const Atom &Atom::At	= atoms[85];
const Atom &Atom::Rn	= atoms[86];
const Atom &Atom::Fr	= atoms[87];
const Atom &Atom::Ra	= atoms[88];
const Atom &Atom::Ac	= atoms[89];
const Atom &Atom::Th	= atoms[90];
const Atom &Atom::Pa	= atoms[91];
const Atom &Atom::U	= atoms[92];
const Atom &Atom::Np	= atoms[93];
const Atom &Atom::Pu	= atoms[94];
const Atom &Atom::Am	= atoms[95];
const Atom &Atom::Cm	= atoms[96];
const Atom &Atom::Bk	= atoms[97];
const Atom &Atom::Cf	= atoms[98];
const Atom &Atom::Es	= atoms[99];
const Atom &Atom::Fm	= atoms[100];
const Atom &Atom::Md	= atoms[101];
const Atom &Atom::No	= atoms[102];
const Atom &Atom::Lr	= atoms[103];
const Atom &Atom::Rf	= atoms[104];
const Atom &Atom::Db	= atoms[105];
const Atom &Atom::Sg	= atoms[106];
const Atom &Atom::Bh	= atoms[107];
const Atom &Atom::Hs	= atoms[108];
const Atom &Atom::Mt	= atoms[109];
const Atom &Atom::Ds	= atoms[110];
const Atom &Atom::Rg	= atoms[111];
const Atom &Atom::Cn	= atoms[112];
const Atom &Atom::Nh	= atoms[113];
const Atom &Atom::Fl	= atoms[114];
const Atom &Atom::Mc	= atoms[115];
const Atom &Atom::Lv	= atoms[116];
const Atom &Atom::Ts	= atoms[117];
const Atom &Atom::Og	= atoms[118];

Atom::Atom(unsigned int atomicNumber){
	this->atomicNumber = atomicNumber;
}

Atom::~Atom(){
}

};	//End of namespace TBTK
