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

/// @cond TBTK_FULL_DOCUMENTATION
#pragma once

//Allow for inclusion in multiple translation units
#ifndef TBTK_NUMPY_INITIALIZING_TRANSLATION_UNIT
#	define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL TBTK_PY_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL TBTK_PY_UFUNC_API

//The convention of putting third party libraries after TBTK and std libraries
//is here ignored. Pyhton.h has to be included first to avoid warning that
//_POSIX_C_SOURCE is redefined.
#include <Python.h>

#include <vector>
#include <map>
#include <array>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstdint> // <cstdint> requires c++11 support
#include <functional>

#ifndef WITHOUT_NUMPY
#	define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#	include <numpy/arrayobject.h>

#	ifdef WITH_OPENCV
#		include <opencv2/opencv.hpp>
#	endif // WITH_OPENCV

/*
 * A bunch of constants were removed in OpenCV 4 in favour of enum classes, so
 * define the ones we need here.
 */
#	if CV_MAJOR_VERSION > 3
#		define CV_BGR2RGB cv::COLOR_BGR2RGB
#		define CV_BGRA2RGBA cv::COLOR_BGRA2RGBA
#	endif
#endif // WITHOUT_NUMPY

#if PY_MAJOR_VERSION >= 3
#	define PyString_FromString PyUnicode_FromString
#	define PyInt_FromLong PyLong_FromLong
#	define PyString_FromString PyUnicode_FromString
#	define PyInt_FromString PyLong_FromString
#endif

namespace TBTK {
namespace Visualization {
namespace MatPlotLib {
namespace matplotlibcpp {
namespace detail {

extern std::string s_backend;

struct Interpreter {
	PyObject *s_python_function_show;
	PyObject *s_python_function_close;
	PyObject *s_python_function_draw;
	PyObject *s_python_function_pause;
	PyObject *s_python_function_save;
	PyObject *s_python_function_figure;
	PyObject *s_python_function_fignum_exists;
	PyObject *s_python_function_plot;
	PyObject *s_python_function_quiver;
	PyObject *s_python_function_semilogx;
	PyObject *s_python_function_semilogy;
	PyObject *s_python_function_loglog;
	PyObject *s_python_function_fill;
	PyObject *s_python_function_fill_between;
	PyObject *s_python_function_hist;
	PyObject *s_python_function_imshow;
	PyObject *s_python_function_scatter;
	PyObject *s_python_function_subplot;
	PyObject *s_python_function_subplot2grid;
	PyObject *s_python_function_legend;
	PyObject *s_python_function_xlim;
	PyObject *s_python_function_ion;
	PyObject *s_python_function_ginput;
	PyObject *s_python_function_ylim;
	PyObject *s_python_function_title;
	PyObject *s_python_function_axis;
	PyObject *s_python_function_xlabel;
	PyObject *s_python_function_ylabel;
	PyObject *s_python_function_xticks;
	PyObject *s_python_function_yticks;
	PyObject *s_python_function_grid;
	PyObject *s_python_function_clf;
	PyObject *s_python_function_errorbar;
	PyObject *s_python_function_annotate;
	PyObject *s_python_function_tight_layout;
	PyObject *s_python_colormap;
	PyObject *s_python_empty_tuple;
	PyObject *s_python_function_stem;
	PyObject *s_python_function_xkcd;
	PyObject *s_python_function_text;
	PyObject *s_python_function_suptitle;
	PyObject *s_python_function_bar;
	PyObject *s_python_function_subplots_adjust;
	//<Added>
	PyObject *s_python_function_gcf;
	PyObject *s_python_function_gca;
	PyObject *mpl_toolkitsmod;
	PyObject *axis3dmod;
	//</Added>

	static Interpreter& get();

	void initializeMPLToolkits();

	PyObject* safe_import(PyObject* module, std::string fname);
private:
#ifndef WITHOUT_NUMPY
#	if PY_MAJOR_VERSION >= 3
	void *import_numpy();
#	else
	void import_numpy();
#	endif
#endif
    Interpreter();

    ~Interpreter();
};

} // end namespace detail
} // end namespace matplotlibcpp
} // end namespace MatPlotLib
} // end namespace Visualization
} // end namespace TBTK
/// @endcond
