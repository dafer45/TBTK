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
#include "TBTK/Visualization/MatPlotLib/matplotlibcpp.h"

namespace matplotlibcpp{
namespace detail{

std::string s_backend;

/* For now, _interpreter is implemented as a singleton since its currently not possible to have
  multiple independent embedded python interpreters without patching the python source code
  or starting a separate process for each.
  http://bytes.com/topic/python/answers/793370-multiple-independent-python-interpreters-c-c-program
 */
_interpreter& _interpreter::get(){
	static _interpreter ctx;
	return ctx;
}

#ifndef WITHOUT_NUMPY
#	if PY_MAJOR_VERSION >= 3
void* _interpreter::import_numpy(){
	import_array();
	return NULL;
}
#	else
void _interpreter::import_numpy(){
	import_array();
}
#	endif
#endif

};	//End of namespace detail
};	//End of namespace matplotlibcpp
