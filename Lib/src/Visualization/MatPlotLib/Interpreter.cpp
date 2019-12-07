/* Copyright 2019 Kristofer Björnson
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

/* The MIT License (MIT)
 *
 * Copyright (c) 2014 Benno Evers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#define TBTK_NUMPY_INITIALIZING_TRANSLATION_UNIT
#include "TBTK/Visualization/MatPlotLib/Interpreter.h"

#include <vector>
#include <map>
#include <array>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstdint> // <cstdint> requires c++11 support
#include <functional>

#include <Python.h>

namespace TBTK {
namespace Visualization {
namespace MatPlotLib {
namespace matplotlibcpp {
namespace detail {

extern std::string s_backend;

#ifdef TBTK_MATPLOTLIB_DO_NOT_FORCE_AGG
	std::string s_backend;
#else
	std::string s_backend = "Agg";
#endif

/* For now, Interpreter is implemented as a singleton since its currently not possible to have
  multiple independent embedded python interpreters without patching the python source code
  or starting a separate process for each.
  http://bytes.com/topic/python/answers/793370-multiple-independent-python-interpreters-c-c-program
 */
Interpreter& Interpreter::get(){
        static Interpreter ctx;
        return ctx;
}

PyObject* Interpreter::safe_import(PyObject* module, std::string fname) {
	PyObject* fn = PyObject_GetAttrString(module, fname.c_str());

	if (!fn)
		throw std::runtime_error(std::string("Couldn't find required function: ") + fname);

	if (!PyFunction_Check(fn))
		throw std::runtime_error(fname + std::string(" is unexpectedly not a PyFunction."));

	return fn;
}

void Interpreter::initializeMPLToolkits(){
	if(!mpl_toolkitsmod){
		PyObject *mpl_toolkits = PyString_FromString("mpl_toolkits");
		PyObject *axis3d = PyString_FromString("mpl_toolkits.mplot3d");
		if(!mpl_toolkits || !axis3d){
			throw std::runtime_error(
				"mpl toolkits: couldn't create string"
			);
		}

		mpl_toolkitsmod = PyImport_Import(mpl_toolkits);
		Py_DECREF(mpl_toolkits);
		if(!mpl_toolkitsmod){
			throw std::runtime_error(
				"Error loading module mpl_toolkits."
			);
		}

		axis3dmod = PyImport_Import(axis3d);
		Py_DECREF(axis3d);
		if(!axis3dmod){
			throw std::runtime_error(
				"Error loading module mpl_toolkits.mplot3d."
			);
		}
	}
}

Interpreter::Interpreter() {
	// optional but recommended
#if PY_MAJOR_VERSION >= 3
	wchar_t name[] = L"plotting";
#else
	char name[] = "plotting";
#endif
	Py_SetProgramName(name);
	Py_Initialize();
#ifndef WITHOUT_NUMPY
	import_numpy(); // initialize numpy C-API
#endif
	PyObject* matplotlibname = PyString_FromString("matplotlib");
	PyObject* pyplotname = PyString_FromString("matplotlib.pyplot");
	PyObject* cmname  = PyString_FromString("matplotlib.cm");
	PyObject* pylabname  = PyString_FromString("pylab");
	if (!pyplotname || !pylabname || !matplotlibname || !cmname) {
		throw std::runtime_error("couldnt create string");
	}

	PyObject* matplotlib = PyImport_Import(matplotlibname);
	Py_DECREF(matplotlibname);
	if (!matplotlib) {
		PyErr_Print();
		throw std::runtime_error("Error loading module matplotlib!");
	}

	// matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
	// or matplotlib.backends is imported for the first time
	if (!s_backend.empty()) {
		PyObject_CallMethod(matplotlib, const_cast<char*>("use"), const_cast<char*>("s"), s_backend.c_str());
	}

	PyObject* pymod = PyImport_Import(pyplotname);
	Py_DECREF(pyplotname);
	if (!pymod) { throw std::runtime_error("Error loading module matplotlib.pyplot!"); }

	s_python_colormap = PyImport_Import(cmname);
	Py_DECREF(cmname);
	if (!s_python_colormap) { throw std::runtime_error("Error loading module matplotlib.cm!"); }

	PyObject* pylabmod = PyImport_Import(pylabname);
	Py_DECREF(pylabname);
	if (!pylabmod) { throw std::runtime_error("Error loading module pylab!"); }

	s_python_function_show = safe_import(pymod, "show");
	s_python_function_close = safe_import(pymod, "close");
	s_python_function_draw = safe_import(pymod, "draw");
	s_python_function_pause = safe_import(pymod, "pause");
	s_python_function_figure = safe_import(pymod, "figure");
	s_python_function_fignum_exists = safe_import(pymod, "fignum_exists");
	s_python_function_plot = safe_import(pymod, "plot");
	s_python_function_quiver = safe_import(pymod, "quiver");
	s_python_function_semilogx = safe_import(pymod, "semilogx");
	s_python_function_semilogy = safe_import(pymod, "semilogy");
	s_python_function_loglog = safe_import(pymod, "loglog");
	s_python_function_fill = safe_import(pymod, "fill");
	s_python_function_fill_between = safe_import(pymod, "fill_between");
	s_python_function_hist = safe_import(pymod,"hist");
	s_python_function_scatter = safe_import(pymod,"scatter");
	s_python_function_subplot = safe_import(pymod, "subplot");
	s_python_function_subplot2grid = safe_import(pymod, "subplot2grid");
	s_python_function_legend = safe_import(pymod, "legend");
	s_python_function_ylim = safe_import(pymod, "ylim");
	s_python_function_title = safe_import(pymod, "title");
	s_python_function_axis = safe_import(pymod, "axis");
	s_python_function_xlabel = safe_import(pymod, "xlabel");
	s_python_function_ylabel = safe_import(pymod, "ylabel");
	s_python_function_xticks = safe_import(pymod, "xticks");
	s_python_function_yticks = safe_import(pymod, "yticks");
	s_python_function_grid = safe_import(pymod, "grid");
	s_python_function_xlim = safe_import(pymod, "xlim");
	s_python_function_ion = safe_import(pymod, "ion");
	s_python_function_ginput = safe_import(pymod, "ginput");
	s_python_function_save = safe_import(pylabmod, "savefig");
	s_python_function_annotate = safe_import(pymod,"annotate");
	s_python_function_clf = safe_import(pymod, "clf");
	s_python_function_errorbar = safe_import(pymod, "errorbar");
	s_python_function_tight_layout = safe_import(pymod, "tight_layout");
	s_python_function_stem = safe_import(pymod, "stem");
	s_python_function_xkcd = safe_import(pymod, "xkcd");
	s_python_function_text = safe_import(pymod, "text");
	s_python_function_suptitle = safe_import(pymod, "suptitle");
	s_python_function_bar = safe_import(pymod,"bar");
	s_python_function_subplots_adjust = safe_import(pymod,"subplots_adjust");
	//<Added>
	s_python_function_gcf = safe_import(pymod, "gcf");
	s_python_function_gca = safe_import(pymod, "gca");
	mpl_toolkitsmod = nullptr;
	axis3dmod = nullptr;
	//</Added>
#ifndef WITHOUT_NUMPY
	s_python_function_imshow = safe_import(pymod, "imshow");
#endif
	s_python_empty_tuple = PyTuple_New(0);
}

Interpreter::~Interpreter() {
	Py_Finalize();
}

#ifndef WITHOUT_NUMPY
#       if PY_MAJOR_VERSION >= 3
void* Interpreter::import_numpy(){
        import_array();
        return NULL;
}
#       else
void Interpreter::import_numpy(){
        import_array();
}
#       endif
#endif

} // end namespace detail
} // end namespace matplotlibcpp
} // end namespace MatPlotLib
} // end namespace Visualization
} // end namespace TBTK
