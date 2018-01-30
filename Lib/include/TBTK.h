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

#include "Model.h"
#include "BlockDiagonalizationSolver.h"
#include "DiagonalizationSolver.h"
#include "ChebyshevSolver.h"
#include "BPropertyExtractor.h"
#include "CPropertyExtractor.h"
#include "DPropertyExtractor.h"
#include "AbstractHoppingAmplitudeFilter.h"
#include "AbstractIndexFilter.h"
#include "Array.h"
#include "ArrayManager.h"
#include "BandDiagramGenerator.h"
#include "FileParser.h"
#include "FileReader.h"
#include "FileWriter.h"
#include "Functions.h"
#include "IndexBasedHoppingAmplitudeFilter.h"
#include "IndexedDataTree.h"
#include "Matrix.h"
#include "MultiCounter.h"
#include "ParameterSet.h"
#include "Range.h"
#include "SerializeableVector.h"
#include "Smooth.h"
#include "SparseMatrix.h"
#include "SpinMatrix.h"
#include "Streams.h"
#include "TBTKMacros.h"
#include "Timer.h"
#include "UnitHandler.h"
#include "Vector2d.h"
#include "Vector3d.h"
#include "VectorNd.h"
#include "WannierParser.h"

#include <complex>

namespace TBTK{

typedef unsigned int Natural;
typedef int Integer;
typedef double Real;
typedef std::complex<double> Complex;

}; //End of namesapce TBTK

using namespace TBTK;

#endif
