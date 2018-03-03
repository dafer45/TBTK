/* Copyright 2018 Kristofer Björnson
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

/** @package TBTKcalc
 *  @file TBTK.h
 *  @brief Header file that can be included in project code to reduce overhead.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_TBTK
#define COM_DAFER45_TBTK_TBTK

#include "TBTK/Model.h"
#include "TBTK/Solver/BlockDiagonalizer.h"
#include "TBTK/Solver/Diagonalizer.h"
#include "TBTK/Solver/ChebyshevExpander.h"
#include "TBTK/PropertyExtractor/BlockDiagonalizer.h"
#include "TBTK/PropertyExtractor/ChebyshevExpander.h"
#include "TBTK/PropertyExtractor/Diagonalizer.h"
#include "TBTK/AbstractHoppingAmplitudeFilter.h"
#include "TBTK/AbstractIndexFilter.h"
#include "TBTK/Array.h"
#include "TBTK/ArrayManager.h"
#include "TBTK/BandDiagramGenerator.h"
#include "TBTK/FileParser.h"
#include "TBTK/FileReader.h"
#include "TBTK/FileWriter.h"
#include "TBTK/Functions.h"
#include "TBTK/IndexBasedHoppingAmplitudeFilter.h"
#include "TBTK/IndexedDataTree.h"
#include "TBTK/Matrix.h"
#include "TBTK/MultiCounter.h"
#include "TBTK/ParameterSet.h"
#include "TBTK/Range.h"
#include "TBTK/SerializeableVector.h"
#include "TBTK/Smooth.h"
#include "TBTK/SparseMatrix.h"
#include "TBTK/SpinMatrix.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"
#include "TBTK/Timer.h"
#include "TBTK/UnitHandler.h"
#include "TBTK/Vector2d.h"
#include "TBTK/Vector3d.h"
#include "TBTK/VectorNd.h"
#include "TBTK/WannierParser.h"

#include <complex>

namespace TBTK{

typedef unsigned int Natural;
typedef int Integer;
typedef double Real;
typedef std::complex<double> Complex;

}; //End of namesapce TBTK

using namespace TBTK;

#endif
