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

#ifndef WITHOUT_NUMPY
#  define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#  include <numpy/arrayobject.h>

#  ifdef WITH_OPENCV
#    include <opencv2/opencv.hpp>
#  endif // WITH_OPENCV

/*
 * A bunch of constants were removed in OpenCV 4 in favour of enum classes, so
 * define the ones we need here.
 */
#  if CV_MAJOR_VERSION > 3
#    define CV_BGR2RGB cv::COLOR_BGR2RGB
#    define CV_BGRA2RGBA cv::COLOR_BGRA2RGBA
#  endif
#endif // WITHOUT_NUMPY

#if PY_MAJOR_VERSION >= 3
#  define PyString_FromString PyUnicode_FromString
#  define PyInt_FromLong PyLong_FromLong
#  define PyString_FromString PyUnicode_FromString
#endif


namespace TBTK {
namespace Visualization {
namespace MatPlotLib {
namespace matplotlibcpp {

// must be called before the first regular call to matplotlib to have any effect
inline void backend(const std::string& name)
{
    detail::s_backend = name;
}

inline bool annotate(std::string annotation, double x, double y)
{
    PyObject * xy = PyTuple_New(2);
    PyObject * str = PyString_FromString(annotation.c_str());

    PyTuple_SetItem(xy,0,PyFloat_FromDouble(x));
    PyTuple_SetItem(xy,1,PyFloat_FromDouble(y));

    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "xy", xy);

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, str);

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_annotate, args, kwargs);

    Py_DECREF(args);
    Py_DECREF(kwargs);

    if(res) Py_DECREF(res);

    return res;
}

#ifndef WITHOUT_NUMPY
// Type selector for numpy array conversion
template <typename T> struct select_npy_type { const static NPY_TYPES type = NPY_NOTYPE; }; //Default
template <> struct select_npy_type<double> { const static NPY_TYPES type = NPY_DOUBLE; };
template <> struct select_npy_type<float> { const static NPY_TYPES type = NPY_FLOAT; };
template <> struct select_npy_type<bool> { const static NPY_TYPES type = NPY_BOOL; };
template <> struct select_npy_type<int8_t> { const static NPY_TYPES type = NPY_INT8; };
template <> struct select_npy_type<int16_t> { const static NPY_TYPES type = NPY_SHORT; };
template <> struct select_npy_type<int32_t> { const static NPY_TYPES type = NPY_INT; };
template <> struct select_npy_type<int64_t> { const static NPY_TYPES type = NPY_INT64; };
template <> struct select_npy_type<uint8_t> { const static NPY_TYPES type = NPY_UINT8; };
template <> struct select_npy_type<uint16_t> { const static NPY_TYPES type = NPY_USHORT; };
template <> struct select_npy_type<uint32_t> { const static NPY_TYPES type = NPY_ULONG; };
template <> struct select_npy_type<uint64_t> { const static NPY_TYPES type = NPY_UINT64; };

template<typename Numeric>
PyObject* get_array(const std::vector<Numeric>& v)
{
    detail::Interpreter::get();    //interpreter needs to be initialized for the numpy commands to work
    NPY_TYPES type = select_npy_type<Numeric>::type;
    if (type == NPY_NOTYPE)
    {
        std::vector<double> vd(v.size());
        npy_intp vsize = v.size();
        std::copy(v.begin(),v.end(),vd.begin());
        PyObject* varray = PyArray_SimpleNewFromData(1, &vsize, NPY_DOUBLE, (void*)(vd.data()));
        return varray;
    }

    npy_intp vsize = v.size();
    PyObject* varray = PyArray_SimpleNewFromData(1, &vsize, type, (void*)(v.data()));
    return varray;
}

template<typename Numeric>
PyObject* get_2darray(const std::vector<::std::vector<Numeric>>& v)
{
    detail::Interpreter::get();    //interpreter needs to be initialized for the numpy commands to work
    if (v.size() < 1) throw std::runtime_error("get_2d_array v too small");

    npy_intp vsize[2] = {static_cast<npy_intp>(v.size()),
                         static_cast<npy_intp>(v[0].size())};

    PyArrayObject *varray =
        (PyArrayObject *)PyArray_SimpleNew(2, vsize, NPY_DOUBLE);

    double *vd_begin = static_cast<double *>(PyArray_DATA(varray));

    for (const ::std::vector<Numeric> &v_row : v) {
      if (v_row.size() != static_cast<size_t>(vsize[1]))
        throw std::runtime_error("Missmatched array size");
      std::copy(v_row.begin(), v_row.end(), vd_begin);
      vd_begin += vsize[1];
    }

    return reinterpret_cast<PyObject *>(varray);
}

template<typename Numeric>
PyObject* get_3darray(const std::vector<std::vector<::std::vector<Numeric>>>& v)
{
    detail::Interpreter::get();    //interpreter needs to be initialized for the numpy commands to work
    if (v.size() < 1) throw std::runtime_error("get_3d_array v too small");
    if (v[0].size() < 1) throw std::runtime_error("get_3d_array v too small");
    for(unsigned int n = 0; n < v.size(); n++)
        if (v[n].size() != v[0].size()) throw std::runtime_error("get_3d_array v too small");

    npy_intp vsize[3] = {
        static_cast<npy_intp>(v.size()),
        static_cast<npy_intp>(v[0].size()),
        static_cast<npy_intp>(v[0][0].size())
    };

    PyArrayObject *varray =
        (PyArrayObject *)PyArray_SimpleNew(3, vsize, NPY_DOUBLE);

    double *vd_begin = static_cast<double *>(PyArray_DATA(varray));

    for (const ::std::vector<::std::vector<Numeric>> &v_x : v) {
      if (v_x.size() != static_cast<size_t>(vsize[1]))
        throw std::runtime_error("Missmatched array size");
      for (const ::std::vector<Numeric> &v_y : v_x) {
        std::copy(v_y.begin(), v_y.end(), vd_begin);
        vd_begin += vsize[2];
      }
    }

    return reinterpret_cast<PyObject *>(varray);
}

#else // fallback if we don't have numpy: copy every element of the given vector

template<typename Numeric>
PyObject* get_array(const std::vector<Numeric>& v)
{
    PyObject* list = PyList_New(v.size());
    for(size_t i = 0; i < v.size(); ++i) {
        PyList_SetItem(list, i, PyFloat_FromDouble(v.at(i)));
    }
    return list;
}

#endif // WITHOUT_NUMPY

template<typename Numeric>
bool plot(const std::vector<Numeric> &x, const std::vector<Numeric> &y, const std::map<std::string, std::string>& keywords)
{
    assert(x.size() == y.size());

    // using numpy arrays
    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    // construct positional args
    PyObject* args = PyTuple_New(2);
    PyTuple_SetItem(args, 0, xarray);
    PyTuple_SetItem(args, 1, yarray);

    // construct keyword args
    PyObject* kwargs = PyDict_New();
    for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        if(it->first.compare("linewidth") == 0){
            char value[it->second.size()+1];
            for(unsigned int n = 0; n < it->second.size(); n++)
                value[n] = it->second[n];
            value[it->second.size()] = '\0';
            PyDict_SetItemString(kwargs, it->first.c_str(), PyInt_FromString(value, nullptr, 0));
        }
        else{
            PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
        }
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_plot, args, kwargs);

    Py_DECREF(args);
    Py_DECREF(kwargs);
    if(res) Py_DECREF(res);

    return res;
}

// TODO - it should be possible to make this work by implementing
// a non-numpy alternative for `get_2darray()`.
#ifndef WITHOUT_NUMPY
template <typename Numeric>
void plot_surface(const std::vector<::std::vector<Numeric>> &x,
                  const std::vector<::std::vector<Numeric>> &y,
                  const std::vector<::std::vector<Numeric>> &z,
                  const std::map<std::string, std::string> &keywords =
                      std::map<std::string, std::string>())
{
  // We lazily load the modules here the first time this function is called
  // because I'm not sure that we can assume "matplotlib installed" implies
  // "mpl_toolkits installed" on all platforms, and we don't want to require
  // it for people who don't need 3d plots.
/*  static PyObject *mpl_toolkitsmod = nullptr, *axis3dmod = nullptr;
  if (!mpl_toolkitsmod) {
    detail::Interpreter::get();

    PyObject* mpl_toolkits = PyString_FromString("mpl_toolkits");
    PyObject* axis3d = PyString_FromString("mpl_toolkits.mplot3d");
    if (!mpl_toolkits || !axis3d) { throw std::runtime_error("couldnt create string"); }

    mpl_toolkitsmod = PyImport_Import(mpl_toolkits);
    Py_DECREF(mpl_toolkits);
    if (!mpl_toolkitsmod) { throw std::runtime_error("Error loading module mpl_toolkits!"); }

    axis3dmod = PyImport_Import(axis3d);
    Py_DECREF(axis3d);
    if (!axis3dmod) { throw std::runtime_error("Error loading module mpl_toolkits.mplot3d!"); }
  }*/
  detail::Interpreter::get().initializeMPLToolkits();

  assert(x.size() == y.size());
  assert(y.size() == z.size());

  // using numpy arrays
  PyObject *xarray = get_2darray(x);
  PyObject *yarray = get_2darray(y);
  PyObject *zarray = get_2darray(z);

  // construct positional args
  PyObject *args = PyTuple_New(3);
  PyTuple_SetItem(args, 0, xarray);
  PyTuple_SetItem(args, 1, yarray);
  PyTuple_SetItem(args, 2, zarray);

  // Build up the kw args.
  PyObject *kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "rstride", PyInt_FromLong(1));
  PyDict_SetItemString(kwargs, "cstride", PyInt_FromLong(1));

  PyObject *python_colormap_coolwarm = PyObject_GetAttrString(
      detail::Interpreter::get().s_python_colormap, "coolwarm");

  PyDict_SetItemString(kwargs, "cmap", python_colormap_coolwarm);

  for (std::map<std::string, std::string>::const_iterator it = keywords.begin();
       it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(),
                         PyString_FromString(it->second.c_str()));
  }


  PyObject *fig =
      PyObject_CallObject(detail::Interpreter::get().s_python_function_figure,
                          detail::Interpreter::get().s_python_empty_tuple);
  if (!fig) throw std::runtime_error("Call to figure() failed.");

  PyObject *gca_kwargs = PyDict_New();
  PyDict_SetItemString(gca_kwargs, "projection", PyString_FromString("3d"));

  PyObject *gca = PyObject_GetAttrString(fig, "gca");
  if (!gca) throw std::runtime_error("No gca");
  Py_INCREF(gca);
  PyObject *axis = PyObject_Call(
      gca, detail::Interpreter::get().s_python_empty_tuple, gca_kwargs);

  if (!axis) throw std::runtime_error("No axis");
  Py_INCREF(axis);

  Py_DECREF(gca);
  Py_DECREF(gca_kwargs);

  PyObject *plot_surface = PyObject_GetAttrString(axis, "plot_surface");
  if (!plot_surface) throw std::runtime_error("No surface");
  Py_INCREF(plot_surface);
  PyObject *res = PyObject_Call(plot_surface, args, kwargs);
  if (!res) throw std::runtime_error("failed surface");
  Py_DECREF(plot_surface);

  Py_DECREF(axis);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  if (res) Py_DECREF(res);
}

template <typename Numeric>
void quiver(
    const std::vector<std::vector<::std::vector<Numeric>>> &x,
    const std::vector<std::vector<::std::vector<Numeric>>> &y,
    const std::vector<std::vector<::std::vector<Numeric>>> &z,
    const std::vector<std::vector<::std::vector<Numeric>>> &u,
    const std::vector<std::vector<::std::vector<Numeric>>> &v,
    const std::vector<std::vector<::std::vector<Numeric>>> &w,
    const std::map<std::string, std::string> &keywords =
    std::map<std::string, std::string>()
){
  // We lazily load the modules here the first time this function is called
  // because I'm not sure that we can assume "matplotlib installed" implies
  // "mpl_toolkits installed" on all platforms, and we don't want to require
  // it for people who don't need 3d plots.
  detail::Interpreter::get().initializeMPLToolkits();

  assert(x.size() == y.size());
  assert(x.size() == z.size());
  assert(x.size() == u.size());
  assert(x.size() == v.size());
  assert(x.size() == z.size());
  for(unsigned int n = 0; n < x.size(); n++){
    assert(x[0].size() == x[n].size());
    assert(x[0].size() == y[n].size());
    assert(x[0].size() == z[n].size());
    assert(x[0].size() == u[n].size());
    assert(x[0].size() == v[n].size());
    assert(x[0].size() == w[n].size());
    for(unsigned int c = 0; c < x[0].size(); c++){
      assert(x[n][0].size() == x[n][c].size());
      assert(x[n][0].size() == y[n][c].size());
      assert(x[n][0].size() == z[n][c].size());
      assert(x[n][0].size() == u[n][c].size());
      assert(x[n][0].size() == v[n][c].size());
      assert(x[n][0].size() == w[n][c].size());
    }
  }

  // using numpy arrays
  PyObject *xarray = get_3darray(x);
  PyObject *yarray = get_3darray(y);
  PyObject *zarray = get_3darray(z);
  PyObject *uarray = get_3darray(u);
  PyObject *varray = get_3darray(v);
  PyObject *warray = get_3darray(w);

  // construct positional args
  PyObject *args = PyTuple_New(6);
  PyTuple_SetItem(args, 0, xarray);
  PyTuple_SetItem(args, 1, yarray);
  PyTuple_SetItem(args, 2, zarray);
  PyTuple_SetItem(args, 3, uarray);
  PyTuple_SetItem(args, 4, varray);
  PyTuple_SetItem(args, 5, warray);

  // Build up the kw args.
  PyObject *kwargs = PyDict_New();
//  PyDict_SetItemString(kwargs, "rstride", PyInt_FromLong(1));
//  PyDict_SetItemString(kwargs, "cstride", PyInt_FromLong(1));

//  PyObject *python_colormap_coolwarm = PyObject_GetAttrString(
//      detail::Interpreter::get().s_python_colormap, "coolwarm");

//  PyDict_SetItemString(kwargs, "cmap", python_colormap_coolwarm);

  for (std::map<std::string, std::string>::const_iterator it = keywords.begin();
       it != keywords.end(); ++it) {
    if(it->first.compare("length") == 0){
      char value[it->second.size()+1];
      for(unsigned int n = 0; n < it->second.size(); n++)
        value[n] = it->second[n];
      value[it->second.size()] = '\0';
      PyDict_SetItemString(
        kwargs,
        it->first.c_str(),
        PyFloat_FromString(PyString_FromString(value))
      );
    }
    else{
      PyDict_SetItemString(
        kwargs,
        it->first.c_str(),
        PyString_FromString(it->second.c_str())
      );
    }
  }


  PyObject *fig =
      PyObject_CallObject(detail::Interpreter::get().s_python_function_figure,
                          detail::Interpreter::get().s_python_empty_tuple);
  if (!fig) throw std::runtime_error("Call to figure() failed.");

  PyObject *gca_kwargs = PyDict_New();
  PyDict_SetItemString(gca_kwargs, "projection", PyString_FromString("3d"));

  PyObject *gca = PyObject_GetAttrString(fig, "gca");
  if (!gca) throw std::runtime_error("No gca");
  Py_INCREF(gca);
  PyObject *axis = PyObject_Call(
      gca, detail::Interpreter::get().s_python_empty_tuple, gca_kwargs);

  if (!axis) throw std::runtime_error("No axis");
  Py_INCREF(axis);

  Py_DECREF(gca);
  Py_DECREF(gca_kwargs);

  PyObject *quiver = PyObject_GetAttrString(axis, "quiver");
  if (!quiver) throw std::runtime_error("No quiver");
  Py_INCREF(quiver);
  PyObject *res = PyObject_Call(quiver, args, kwargs);
  if (!res) throw std::runtime_error("failed quiver");
  Py_DECREF(quiver);

  Py_DECREF(axis);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  if (res) Py_DECREF(res);
}

//template <typename Numeric>
inline void view_init(
  const std::map<std::string, std::string> &keywords =
    std::map<std::string, std::string>())
{
  PyObject *axis = PyObject_CallObject(
    detail::Interpreter::get().s_python_function_gca,
    detail::Interpreter::get().s_python_empty_tuple
  );

  PyObject *view_init_kwargs = PyDict_New();
  std::string elev = keywords.at("elev");
  std::string azim = keywords.at("azim");
  PyDict_SetItemString(view_init_kwargs, "elev", PyInt_FromString((char*)elev.c_str(), nullptr, 0));
  PyDict_SetItemString(view_init_kwargs, "azim", PyInt_FromString((char*)azim.c_str(), nullptr, 0));

  PyObject *view_init = PyObject_GetAttrString(axis, "view_init");
  if (!view_init) throw std::runtime_error("No view_init");
  Py_INCREF(view_init);
  PyObject *res = PyObject_Call(
    view_init,
    detail::Interpreter::get().s_python_empty_tuple,
    view_init_kwargs
  );
  if (!res) throw std::runtime_error("failed view_init");
  Py_DECREF(view_init);
  if (res) Py_DECREF(res);

  Py_DECREF(axis);
}
#endif // WITHOUT_NUMPY

template <typename Numeric>
void contourf(
  const std::vector<::std::vector<Numeric>> &x,
  const std::vector<::std::vector<Numeric>> &y,
  const std::vector<::std::vector<Numeric>> &z,
  unsigned int levels = 8,
  const std::map<std::string, std::string> &keywords =
    std::map<std::string, std::string>()
){
  assert(x.size() == y.size());
  assert(y.size() == z.size());

  // using numpy arrays
  PyObject *xarray = get_2darray(x);
  PyObject *yarray = get_2darray(y);
  PyObject *zarray = get_2darray(z);

  // construct positional args
  PyObject *args = PyTuple_New(4);
  PyTuple_SetItem(args, 0, xarray);
  PyTuple_SetItem(args, 1, yarray);
  PyTuple_SetItem(args, 2, zarray);
  PyTuple_SetItem(args, 3, PyInt_FromLong(levels));

  // Build up the kw args.
  PyObject *kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "rstride", PyInt_FromLong(1));
  PyDict_SetItemString(kwargs, "cstride", PyInt_FromLong(1));

  PyObject *python_colormap_coolwarm = PyObject_GetAttrString(
      detail::Interpreter::get().s_python_colormap, "coolwarm");

  PyDict_SetItemString(kwargs, "cmap", python_colormap_coolwarm);

  for (std::map<std::string, std::string>::const_iterator it = keywords.begin();
       it != keywords.end(); ++it) {
    PyDict_SetItemString(kwargs, it->first.c_str(),
                         PyString_FromString(it->second.c_str()));
  }


  PyObject *fig =
      PyObject_CallObject(detail::Interpreter::get().s_python_function_figure,
                          detail::Interpreter::get().s_python_empty_tuple);
  if (!fig) throw std::runtime_error("Call to figure() failed.");

  PyObject *gca_kwargs = PyDict_New();
//  PyDict_SetItemString(gca_kwargs, "projection", PyString_FromString("3d"));

  PyObject *gca = PyObject_GetAttrString(fig, "gca");
  if (!gca) throw std::runtime_error("No gca");
  Py_INCREF(gca);
  PyObject *axis = PyObject_Call(
      gca, detail::Interpreter::get().s_python_empty_tuple, gca_kwargs);

  if (!axis) throw std::runtime_error("No axis");
  Py_INCREF(axis);

  Py_DECREF(gca);
  Py_DECREF(gca_kwargs);

  PyObject *contourf = PyObject_GetAttrString(axis, "contourf");
  if (!contourf) throw std::runtime_error("No surface");
  Py_INCREF(contourf);
  PyObject *res = PyObject_Call(contourf, args, kwargs);
  if (!res) throw std::runtime_error("failed surface");

  PyObject *colorbar_kwargs = PyDict_New();
  PyDict_SetItemString(colorbar_kwargs, "mappable", res);
  PyDict_SetItemString(colorbar_kwargs, "ax", axis);
  PyObject *colorbar = PyObject_GetAttrString(fig, "colorbar");
  if(!colorbar) throw std::runtime_error("No colorbar");
  Py_INCREF(colorbar);
  PyObject *res2 = PyObject_Call(
    colorbar,
    detail::Interpreter::get().s_python_empty_tuple,
    colorbar_kwargs
  );
  if(!res2) throw std::runtime_error("failed colorbar");
  Py_DECREF(res2);

  Py_DECREF(contourf);


  Py_DECREF(axis);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  if (res) Py_DECREF(res);
}

template<typename Numeric>
bool stem(const std::vector<Numeric> &x, const std::vector<Numeric> &y, const std::map<std::string, std::string>& keywords)
{
    assert(x.size() == y.size());

    // using numpy arrays
    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    // construct positional args
    PyObject* args = PyTuple_New(2);
    PyTuple_SetItem(args, 0, xarray);
    PyTuple_SetItem(args, 1, yarray);

    // construct keyword args
    PyObject* kwargs = PyDict_New();
    for (std::map<std::string, std::string>::const_iterator it =
            keywords.begin(); it != keywords.end(); ++it) {
        PyDict_SetItemString(kwargs, it->first.c_str(),
                PyString_FromString(it->second.c_str()));
    }

    PyObject* res = PyObject_Call(
            detail::Interpreter::get().s_python_function_stem, args, kwargs);

    Py_DECREF(args);
    Py_DECREF(kwargs);
    if (res)
        Py_DECREF(res);

    return res;
}

template< typename Numeric >
bool fill(const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::map<std::string, std::string>& keywords)
{
    assert(x.size() == y.size());

    // using numpy arrays
    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    // construct positional args
    PyObject* args = PyTuple_New(2);
    PyTuple_SetItem(args, 0, xarray);
    PyTuple_SetItem(args, 1, yarray);

    // construct keyword args
    PyObject* kwargs = PyDict_New();
    for (auto it = keywords.begin(); it != keywords.end(); ++it) {
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_fill, args, kwargs);

    Py_DECREF(args);
    Py_DECREF(kwargs);

    if (res) Py_DECREF(res);

    return res;
}

template< typename Numeric >
bool fill_between(const std::vector<Numeric>& x, const std::vector<Numeric>& y1, const std::vector<Numeric>& y2, const std::map<std::string, std::string>& keywords)
{
    assert(x.size() == y1.size());
    assert(x.size() == y2.size());

    // using numpy arrays
    PyObject* xarray = get_array(x);
    PyObject* y1array = get_array(y1);
    PyObject* y2array = get_array(y2);

    // construct positional args
    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, xarray);
    PyTuple_SetItem(args, 1, y1array);
    PyTuple_SetItem(args, 2, y2array);

    // construct keyword args
    PyObject* kwargs = PyDict_New();
    for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it) {
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_fill_between, args, kwargs);

    Py_DECREF(args);
    Py_DECREF(kwargs);
    if(res) Py_DECREF(res);

    return res;
}

template< typename Numeric>
bool hist(const std::vector<Numeric>& y, long bins=10,std::string color="b",
          double alpha=1.0, bool cumulative=false)
{

    PyObject* yarray = get_array(y);

    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "bins", PyLong_FromLong(bins));
    PyDict_SetItemString(kwargs, "color", PyString_FromString(color.c_str()));
    PyDict_SetItemString(kwargs, "alpha", PyFloat_FromDouble(alpha));
    PyDict_SetItemString(kwargs, "cumulative", cumulative ? Py_True : Py_False);

    PyObject* plot_args = PyTuple_New(1);

    PyTuple_SetItem(plot_args, 0, yarray);


    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_hist, plot_args, kwargs);


    Py_DECREF(plot_args);
    Py_DECREF(kwargs);
    if(res) Py_DECREF(res);

    return res;
}

#ifndef WITHOUT_NUMPY
    namespace internal {
        inline void imshow(void *ptr, const NPY_TYPES type, const int rows, const int columns, const int colors, const std::map<std::string, std::string> &keywords)
        {
            assert(type == NPY_UINT8 || type == NPY_FLOAT);
            assert(colors == 1 || colors == 3 || colors == 4);

            detail::Interpreter::get();    //interpreter needs to be initialized for the numpy commands to work

            // construct args
            npy_intp dims[3] = { rows, columns, colors };
            PyObject *args = PyTuple_New(1);
            PyTuple_SetItem(args, 0, PyArray_SimpleNewFromData(colors == 1 ? 2 : 3, dims, type, ptr));

            // construct keyword args
            PyObject* kwargs = PyDict_New();
            for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
            {
                PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
            }

            PyObject *res = PyObject_Call(detail::Interpreter::get().s_python_function_imshow, args, kwargs);
            Py_DECREF(args);
            Py_DECREF(kwargs);
            if (!res)
                throw std::runtime_error("Call to imshow() failed");
            Py_DECREF(res);
        }
    }

    inline void imshow(const unsigned char *ptr, const int rows, const int columns, const int colors, const std::map<std::string, std::string> &keywords = {})
    {
        internal::imshow((void *) ptr, NPY_UINT8, rows, columns, colors, keywords);
    }

    inline void imshow(const float *ptr, const int rows, const int columns, const int colors, const std::map<std::string, std::string> &keywords = {})
    {
        internal::imshow((void *) ptr, NPY_FLOAT, rows, columns, colors, keywords);
    }

#ifdef WITH_OPENCV
    void imshow(const cv::Mat &image, const std::map<std::string, std::string> &keywords = {})
    {
        // Convert underlying type of matrix, if needed
        cv::Mat image2;
        NPY_TYPES npy_type = NPY_UINT8;
        switch (image.type() & CV_MAT_DEPTH_MASK) {
        case CV_8U:
            image2 = image;
            break;
        case CV_32F:
            image2 = image;
            npy_type = NPY_FLOAT;
            break;
        default:
            image.convertTo(image2, CV_MAKETYPE(CV_8U, image.channels()));
        }

        // If color image, convert from BGR to RGB
        switch (image2.channels()) {
        case 3:
            cv::cvtColor(image2, image2, CV_BGR2RGB);
            break;
        case 4:
            cv::cvtColor(image2, image2, CV_BGRA2RGBA);
        }

        internal::imshow(image2.data, npy_type, image2.rows, image2.cols, image2.channels(), keywords);
    }
#endif // WITH_OPENCV
#endif // WITHOUT_NUMPY

template<typename NumericX, typename NumericY>
bool scatter(const std::vector<NumericX>& x,
             const std::vector<NumericY>& y,
             const double s=1.0) // The marker size in points**2
{
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "s", PyLong_FromLong(s));

    PyObject* plot_args = PyTuple_New(2);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_scatter, plot_args, kwargs);

    Py_DECREF(plot_args);
    Py_DECREF(kwargs);
    if(res) Py_DECREF(res);

    return res;
}

template <typename Numeric>
bool bar(const std::vector<Numeric> &               x,
         const std::vector<Numeric> &               y,
         std::string                                ec       = "black",
         std::string                                ls       = "-",
         double                                     lw       = 1.0,
         const std::map<std::string, std::string> & keywords = {}) {
  PyObject * xarray = get_array(x);
  PyObject * yarray = get_array(y);

  PyObject * kwargs = PyDict_New();

  PyDict_SetItemString(kwargs, "ec", PyString_FromString(ec.c_str()));
  PyDict_SetItemString(kwargs, "ls", PyString_FromString(ls.c_str()));
  PyDict_SetItemString(kwargs, "lw", PyFloat_FromDouble(lw));

  for (std::map<std::string, std::string>::const_iterator it =
         keywords.begin();
       it != keywords.end();
       ++it) {
    PyDict_SetItemString(
      kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
  }

  PyObject * plot_args = PyTuple_New(2);
  PyTuple_SetItem(plot_args, 0, xarray);
  PyTuple_SetItem(plot_args, 1, yarray);

  PyObject * res = PyObject_Call(
    detail::Interpreter::get().s_python_function_bar, plot_args, kwargs);

  Py_DECREF(plot_args);
  Py_DECREF(kwargs);
  if (res) Py_DECREF(res);

  return res;
}

template <typename Numeric>
bool bar(const std::vector<Numeric> &               y,
         std::string                                ec       = "black",
         std::string                                ls       = "-",
         double                                     lw       = 1.0,
         const std::map<std::string, std::string> & keywords = {}) {
  using T = typename std::remove_reference<decltype(y)>::type::value_type;

  std::vector<T> x;
  for (std::size_t i = 0; i < y.size(); i++) { x.push_back(i); }

  return bar(x, y, ec, ls, lw, keywords);
}

inline bool subplots_adjust(const std::map<std::string, double>& keywords = {})
{

    PyObject* kwargs = PyDict_New();
    for (std::map<std::string, double>::const_iterator it =
            keywords.begin(); it != keywords.end(); ++it) {
        PyDict_SetItemString(kwargs, it->first.c_str(),
                             PyFloat_FromDouble(it->second));
    }


    PyObject* plot_args = PyTuple_New(0);

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_subplots_adjust, plot_args, kwargs);

    Py_DECREF(plot_args);
    Py_DECREF(kwargs);
    if(res) Py_DECREF(res);

    return res;
}

template< typename Numeric>
bool named_hist(std::string label,const std::vector<Numeric>& y, long bins=10, std::string color="b", double alpha=1.0)
{
    PyObject* yarray = get_array(y);

    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "label", PyString_FromString(label.c_str()));
    PyDict_SetItemString(kwargs, "bins", PyLong_FromLong(bins));
    PyDict_SetItemString(kwargs, "color", PyString_FromString(color.c_str()));
    PyDict_SetItemString(kwargs, "alpha", PyFloat_FromDouble(alpha));


    PyObject* plot_args = PyTuple_New(1);
    PyTuple_SetItem(plot_args, 0, yarray);

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_hist, plot_args, kwargs);

    Py_DECREF(plot_args);
    Py_DECREF(kwargs);
    if(res) Py_DECREF(res);

    return res;
}

template<typename NumericX, typename NumericY>
bool plot(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "")
{
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(s.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_plot, plot_args);

    Py_DECREF(plot_args);
    if(res) Py_DECREF(res);

    return res;
}

template<typename NumericX, typename NumericY, typename NumericU, typename NumericW>
bool quiver(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::vector<NumericU>& u, const std::vector<NumericW>& w, const std::map<std::string, std::string>& keywords = {})
{
    assert(x.size() == y.size() && x.size() == u.size() && u.size() == w.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);
    PyObject* uarray = get_array(u);
    PyObject* warray = get_array(w);

    PyObject* plot_args = PyTuple_New(4);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, uarray);
    PyTuple_SetItem(plot_args, 3, warray);

    // construct keyword args
    PyObject* kwargs = PyDict_New();
    for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject* res = PyObject_Call(
            detail::Interpreter::get().s_python_function_quiver, plot_args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(plot_args);
    if (res)
        Py_DECREF(res);

    return res;
}

template<typename NumericX, typename NumericY>
bool stem(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "")
{
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(s.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_CallObject(
            detail::Interpreter::get().s_python_function_stem, plot_args);

    Py_DECREF(plot_args);
    if (res)
        Py_DECREF(res);

    return res;
}

template<typename NumericX, typename NumericY>
bool semilogx(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "")
{
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(s.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_semilogx, plot_args);

    Py_DECREF(plot_args);
    if(res) Py_DECREF(res);

    return res;
}

template<typename NumericX, typename NumericY>
bool semilogy(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "")
{
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(s.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_semilogy, plot_args);

    Py_DECREF(plot_args);
    if(res) Py_DECREF(res);

    return res;
}

template<typename NumericX, typename NumericY>
bool loglog(const std::vector<NumericX>& x, const std::vector<NumericY>& y, const std::string& s = "")
{
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(s.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_loglog, plot_args);

    Py_DECREF(plot_args);
    if(res) Py_DECREF(res);

    return res;
}

template<typename NumericX, typename NumericY>
bool errorbar(const std::vector<NumericX> &x, const std::vector<NumericY> &y, const std::vector<NumericX> &yerr, const std::map<std::string, std::string> &keywords = {})
{
    assert(x.size() == y.size());

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);
    PyObject* yerrarray = get_array(yerr);

    // construct keyword args
    PyObject* kwargs = PyDict_New();
    for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }

    PyDict_SetItemString(kwargs, "yerr", yerrarray);

    PyObject *plot_args = PyTuple_New(2);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);

    PyObject *res = PyObject_Call(detail::Interpreter::get().s_python_function_errorbar, plot_args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(plot_args);

    if (res)
        Py_DECREF(res);
    else
        throw std::runtime_error("Call to errorbar() failed.");

    return res;
}

template<typename Numeric>
bool named_plot(const std::string& name, const std::vector<Numeric>& y, const std::string& format = "")
{
    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* plot_args = PyTuple_New(2);

    PyTuple_SetItem(plot_args, 0, yarray);
    PyTuple_SetItem(plot_args, 1, pystring);

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_plot, plot_args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(plot_args);
    if (res) Py_DECREF(res);

    return res;
}

template<typename Numeric>
bool named_plot(const std::string& name, const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::string& format = "")
{
    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_plot, plot_args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(plot_args);
    if (res) Py_DECREF(res);

    return res;
}

template<typename Numeric>
bool named_semilogx(const std::string& name, const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::string& format = "")
{
    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_semilogx, plot_args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(plot_args);
    if (res) Py_DECREF(res);

    return res;
}

template<typename Numeric>
bool named_semilogy(const std::string& name, const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::string& format = "")
{
    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_semilogy, plot_args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(plot_args);
    if (res) Py_DECREF(res);

    return res;
}

template<typename Numeric>
bool named_loglog(const std::string& name, const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::string& format = "")
{
    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

    PyObject* xarray = get_array(x);
    PyObject* yarray = get_array(y);

    PyObject* pystring = PyString_FromString(format.c_str());

    PyObject* plot_args = PyTuple_New(3);
    PyTuple_SetItem(plot_args, 0, xarray);
    PyTuple_SetItem(plot_args, 1, yarray);
    PyTuple_SetItem(plot_args, 2, pystring);
    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_loglog, plot_args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(plot_args);
    if (res) Py_DECREF(res);

    return res;
}

template<typename Numeric>
bool plot(const std::vector<Numeric>& y, const std::string& format = "")
{
    std::vector<Numeric> x(y.size());
    for(size_t i=0; i<x.size(); ++i) x.at(i) = i;
    return plot(x,y,format);
}

template<typename Numeric>
bool plot(const std::vector<Numeric>& y, const std::map<std::string, std::string>& keywords)
{
    std::vector<Numeric> x(y.size());
    for(size_t i=0; i<x.size(); ++i) x.at(i) = i;
    return plot(x,y,keywords);
}

template<typename Numeric>
bool stem(const std::vector<Numeric>& y, const std::string& format = "")
{
    std::vector<Numeric> x(y.size());
    for (size_t i = 0; i < x.size(); ++i) x.at(i) = i;
    return stem(x, y, format);
}

template<typename Numeric>
void text(Numeric x, Numeric y, const std::string& s = "")
{
    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, PyFloat_FromDouble(x));
    PyTuple_SetItem(args, 1, PyFloat_FromDouble(y));
    PyTuple_SetItem(args, 2, PyString_FromString(s.c_str()));

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_text, args);
    if(!res) throw std::runtime_error("Call to text() failed.");

    Py_DECREF(args);
    Py_DECREF(res);
}


inline long figure(long number = -1)
{
    PyObject *res;
    if (number == -1)
        res = PyObject_CallObject(detail::Interpreter::get().s_python_function_figure, detail::Interpreter::get().s_python_empty_tuple);
    else {
        assert(number > 0);

        // Make sure interpreter is initialised
        detail::Interpreter::get();

        PyObject *args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, PyLong_FromLong(number));
        res = PyObject_CallObject(detail::Interpreter::get().s_python_function_figure, args);
        Py_DECREF(args);
    }

    if(!res) throw std::runtime_error("Call to figure() failed.");

    PyObject* num = PyObject_GetAttrString(res, "number");
    if (!num) throw std::runtime_error("Could not get number attribute of figure object");
    const long figureNumber = PyLong_AsLong(num);

    Py_DECREF(num);
    Py_DECREF(res);

    return figureNumber;
}

inline bool fignum_exists(long number)
{
    // Make sure interpreter is initialised
    detail::Interpreter::get();

    PyObject *args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyLong_FromLong(number));
    PyObject *res = PyObject_CallObject(detail::Interpreter::get().s_python_function_fignum_exists, args);
    if(!res) throw std::runtime_error("Call to fignum_exists() failed.");

    bool ret = PyObject_IsTrue(res);
    Py_DECREF(res);
    Py_DECREF(args);

    return ret;
}

inline void figure_size(size_t w, size_t h)
{
    // Make sure interpreter is initialised
    detail::Interpreter::get();

    const size_t dpi = 100;
    PyObject* size = PyTuple_New(2);
    PyTuple_SetItem(size, 0, PyFloat_FromDouble((double)w / dpi));
    PyTuple_SetItem(size, 1, PyFloat_FromDouble((double)h / dpi));

    PyObject* kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "figsize", size);
    PyDict_SetItemString(kwargs, "dpi", PyLong_FromSize_t(dpi));

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_figure,
            detail::Interpreter::get().s_python_empty_tuple, kwargs);

    Py_DECREF(kwargs);

    if(!res) throw std::runtime_error("Call to figure_size() failed.");
    Py_DECREF(res);
}

inline void legend()
{
    PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_legend, detail::Interpreter::get().s_python_empty_tuple);
    if(!res) throw std::runtime_error("Call to legend() failed.");

    Py_DECREF(res);
}

template<typename Numeric>
void ylim(Numeric left, Numeric right)
{
    PyObject* list = PyList_New(2);
    PyList_SetItem(list, 0, PyFloat_FromDouble(left));
    PyList_SetItem(list, 1, PyFloat_FromDouble(right));

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, list);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_ylim, args);
    if(!res) throw std::runtime_error("Call to ylim() failed.");

    Py_DECREF(args);
    Py_DECREF(res);
}

template<typename Numeric>
void xlim(Numeric left, Numeric right)
{
    PyObject* list = PyList_New(2);
    PyList_SetItem(list, 0, PyFloat_FromDouble(left));
    PyList_SetItem(list, 1, PyFloat_FromDouble(right));

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, list);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_xlim, args);
    if(!res) throw std::runtime_error("Call to xlim() failed.");

    Py_DECREF(args);
    Py_DECREF(res);
}


inline double* xlim()
{
    PyObject* args = PyTuple_New(0);
    PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_xlim, args);
    PyObject* left = PyTuple_GetItem(res,0);
    PyObject* right = PyTuple_GetItem(res,1);

    double* arr = new double[2];
    arr[0] = PyFloat_AsDouble(left);
    arr[1] = PyFloat_AsDouble(right);

    if(!res) throw std::runtime_error("Call to xlim() failed.");

    Py_DECREF(res);
    return arr;
}


inline double* ylim()
{
    PyObject* args = PyTuple_New(0);
    PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_ylim, args);
    PyObject* left = PyTuple_GetItem(res,0);
    PyObject* right = PyTuple_GetItem(res,1);

    double* arr = new double[2];
    arr[0] = PyFloat_AsDouble(left);
    arr[1] = PyFloat_AsDouble(right);

    if(!res) throw std::runtime_error("Call to ylim() failed.");

    Py_DECREF(res);
    return arr;
}

template<typename Numeric>
inline void xticks(const std::vector<Numeric> &ticks, const std::vector<std::string> &labels = {}, const std::map<std::string, std::string>& keywords = {})
{
    assert(labels.size() == 0 || ticks.size() == labels.size());

    // using numpy array
    PyObject* ticksarray = get_array(ticks);

    PyObject* args;
    if(labels.size() == 0) {
        // construct positional args
        args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, ticksarray);
    } else {
        // make tuple of tick labels
        PyObject* labelstuple = PyTuple_New(labels.size());
        for (size_t i = 0; i < labels.size(); i++)
            PyTuple_SetItem(labelstuple, i, PyUnicode_FromString(labels[i].c_str()));

        // construct positional args
        args = PyTuple_New(2);
        PyTuple_SetItem(args, 0, ticksarray);
        PyTuple_SetItem(args, 1, labelstuple);
    }

    // construct keyword args
    PyObject* kwargs = PyDict_New();
    for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_xticks, args, kwargs);

    Py_DECREF(args);
    Py_DECREF(kwargs);
    if(!res) throw std::runtime_error("Call to xticks() failed");

    Py_DECREF(res);
}

template<typename Numeric>
inline void xticks(const std::vector<Numeric> &ticks, const std::map<std::string, std::string>& keywords)
{
    xticks(ticks, {}, keywords);
}

template<typename Numeric>
inline void yticks(const std::vector<Numeric> &ticks, const std::vector<std::string> &labels = {}, const std::map<std::string, std::string>& keywords = {})
{
    assert(labels.size() == 0 || ticks.size() == labels.size());

    // using numpy array
    PyObject* ticksarray = get_array(ticks);

    PyObject* args;
    if(labels.size() == 0) {
        // construct positional args
        args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, ticksarray);
    } else {
        // make tuple of tick labels
        PyObject* labelstuple = PyTuple_New(labels.size());
        for (size_t i = 0; i < labels.size(); i++)
            PyTuple_SetItem(labelstuple, i, PyUnicode_FromString(labels[i].c_str()));

        // construct positional args
        args = PyTuple_New(2);
        PyTuple_SetItem(args, 0, ticksarray);
        PyTuple_SetItem(args, 1, labelstuple);
    }

    // construct keyword args
    PyObject* kwargs = PyDict_New();
    for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        PyDict_SetItemString(kwargs, it->first.c_str(), PyString_FromString(it->second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_yticks, args, kwargs);

    Py_DECREF(args);
    Py_DECREF(kwargs);
    if(!res) throw std::runtime_error("Call to yticks() failed");

    Py_DECREF(res);
}

template<typename Numeric>
inline void yticks(const std::vector<Numeric> &ticks, const std::map<std::string, std::string>& keywords)
{
    yticks(ticks, {}, keywords);
}

inline void subplot(long nrows, long ncols, long plot_number)
{
    // construct positional args
    PyObject* args = PyTuple_New(3);
    PyTuple_SetItem(args, 0, PyFloat_FromDouble(nrows));
    PyTuple_SetItem(args, 1, PyFloat_FromDouble(ncols));
    PyTuple_SetItem(args, 2, PyFloat_FromDouble(plot_number));

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_subplot, args);
    if(!res) throw std::runtime_error("Call to subplot() failed.");

    Py_DECREF(args);
    Py_DECREF(res);
}

inline void subplot2grid(long nrows, long ncols, long rowid=0, long colid=0, long rowspan=1, long colspan=1)
{
    PyObject* shape = PyTuple_New(2);
    PyTuple_SetItem(shape, 0, PyLong_FromLong(nrows));
    PyTuple_SetItem(shape, 1, PyLong_FromLong(ncols));

    PyObject* loc = PyTuple_New(2);
    PyTuple_SetItem(loc, 0, PyLong_FromLong(rowid));
    PyTuple_SetItem(loc, 1, PyLong_FromLong(colid));

    PyObject* args = PyTuple_New(4);
    PyTuple_SetItem(args, 0, shape);
    PyTuple_SetItem(args, 1, loc);
    PyTuple_SetItem(args, 2, PyLong_FromLong(rowspan));
    PyTuple_SetItem(args, 3, PyLong_FromLong(colspan));

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_subplot2grid, args);
    if(!res) throw std::runtime_error("Call to subplot2grid() failed.");

    Py_DECREF(shape);
    Py_DECREF(loc);
    Py_DECREF(args);
    Py_DECREF(res);
}

inline void title(const std::string &titlestr, const std::map<std::string, std::string> &keywords = {})
{
    PyObject* pytitlestr = PyString_FromString(titlestr.c_str());
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pytitlestr);

    PyObject* kwargs = PyDict_New();
    for (auto it = keywords.begin(); it != keywords.end(); ++it) {
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_title, args, kwargs);
    if(!res) throw std::runtime_error("Call to title() failed.");

    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(res);
}

inline void suptitle(const std::string &suptitlestr, const std::map<std::string, std::string> &keywords = {})
{
    PyObject* pysuptitlestr = PyString_FromString(suptitlestr.c_str());
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pysuptitlestr);

    PyObject* kwargs = PyDict_New();
    for (auto it = keywords.begin(); it != keywords.end(); ++it) {
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_suptitle, args, kwargs);
    if(!res) throw std::runtime_error("Call to suptitle() failed.");

    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(res);
}

inline void axis(const std::string &axisstr)
{
    PyObject* str = PyString_FromString(axisstr.c_str());
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, str);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_axis, args);
    if(!res) throw std::runtime_error("Call to title() failed.");

    Py_DECREF(args);
    Py_DECREF(res);
}

inline void xlabel(const std::string &str, const std::map<std::string, std::string> &keywords = {})
{
    PyObject* pystr = PyString_FromString(str.c_str());
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pystr);

    PyObject* kwargs = PyDict_New();
    for (auto it = keywords.begin(); it != keywords.end(); ++it) {
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_xlabel, args, kwargs);
    if(!res) throw std::runtime_error("Call to xlabel() failed.");

    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(res);
}

inline void ylabel(const std::string &str, const std::map<std::string, std::string>& keywords = {})
{
    PyObject* pystr = PyString_FromString(str.c_str());
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pystr);

    PyObject* kwargs = PyDict_New();
    for (auto it = keywords.begin(); it != keywords.end(); ++it) {
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_ylabel, args, kwargs);
    if(!res) throw std::runtime_error("Call to ylabel() failed.");

    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(res);
}

inline void grid(bool flag)
{
    PyObject* pyflag = flag ? Py_True : Py_False;
    Py_INCREF(pyflag);

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pyflag);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_grid, args);
    if(!res) throw std::runtime_error("Call to grid() failed.");

    Py_DECREF(args);
    Py_DECREF(res);
}

inline void show(const bool block = true)
{
    PyObject* res;
    if(block)
    {
        res = PyObject_CallObject(
                detail::Interpreter::get().s_python_function_show,
                detail::Interpreter::get().s_python_empty_tuple);
    }
    else
    {
        PyObject *kwargs = PyDict_New();
        PyDict_SetItemString(kwargs, "block", Py_False);
        res = PyObject_Call( detail::Interpreter::get().s_python_function_show, detail::Interpreter::get().s_python_empty_tuple, kwargs);
       Py_DECREF(kwargs);
    }


    if (!res) throw std::runtime_error("Call to show() failed.");

    Py_DECREF(res);
}

inline void close()
{
    PyObject* res = PyObject_CallObject(
            detail::Interpreter::get().s_python_function_close,
            detail::Interpreter::get().s_python_empty_tuple);

    if (!res) throw std::runtime_error("Call to close() failed.");

    Py_DECREF(res);
}

inline void xkcd() {
    PyObject* res;
    PyObject *kwargs = PyDict_New();

    res = PyObject_Call(detail::Interpreter::get().s_python_function_xkcd,
            detail::Interpreter::get().s_python_empty_tuple, kwargs);

    Py_DECREF(kwargs);

    if (!res)
        throw std::runtime_error("Call to show() failed.");

    Py_DECREF(res);
}

inline void draw()
{
    PyObject* res = PyObject_CallObject(
        detail::Interpreter::get().s_python_function_draw,
        detail::Interpreter::get().s_python_empty_tuple);

    if (!res) throw std::runtime_error("Call to draw() failed.");

    Py_DECREF(res);
}

template<typename Numeric>
inline void pause(Numeric interval)
{
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyFloat_FromDouble(interval));

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_pause, args);
    if(!res) throw std::runtime_error("Call to pause() failed.");

    Py_DECREF(args);
    Py_DECREF(res);
}

inline void save(const std::string& filename)
{
    PyObject* pyfilename = PyString_FromString(filename.c_str());

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, pyfilename);

    PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_save, args);
    if (!res) throw std::runtime_error("Call to save() failed.");

    Py_DECREF(args);
    Py_DECREF(res);
}

inline void clf() {
    PyObject *res = PyObject_CallObject(
        detail::Interpreter::get().s_python_function_clf,
        detail::Interpreter::get().s_python_empty_tuple);

    if (!res) throw std::runtime_error("Call to clf() failed.");

    Py_DECREF(res);
}

    inline void ion() {
    PyObject *res = PyObject_CallObject(
        detail::Interpreter::get().s_python_function_ion,
        detail::Interpreter::get().s_python_empty_tuple);

    if (!res) throw std::runtime_error("Call to ion() failed.");

    Py_DECREF(res);
}

inline std::vector<std::array<double, 2>> ginput(const int numClicks = 1, const std::map<std::string, std::string>& keywords = {})
{
    PyObject *args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyLong_FromLong(numClicks));

    // construct keyword args
    PyObject* kwargs = PyDict_New();
    for(std::map<std::string, std::string>::const_iterator it = keywords.begin(); it != keywords.end(); ++it)
    {
        PyDict_SetItemString(kwargs, it->first.c_str(), PyUnicode_FromString(it->second.c_str()));
    }

    PyObject* res = PyObject_Call(
        detail::Interpreter::get().s_python_function_ginput, args, kwargs);

    Py_DECREF(kwargs);
    Py_DECREF(args);
    if (!res) throw std::runtime_error("Call to ginput() failed.");

    const size_t len = PyList_Size(res);
    std::vector<std::array<double, 2>> out;
    out.reserve(len);
    for (size_t i = 0; i < len; i++) {
        PyObject *current = PyList_GetItem(res, i);
        std::array<double, 2> position;
        position[0] = PyFloat_AsDouble(PyTuple_GetItem(current, 0));
        position[1] = PyFloat_AsDouble(PyTuple_GetItem(current, 1));
        out.push_back(position);
    }
    Py_DECREF(res);

    return out;
}

// Actually, is there any reason not to call this automatically for every plot?
inline void tight_layout() {
    PyObject *res = PyObject_CallObject(
        detail::Interpreter::get().s_python_function_tight_layout,
        detail::Interpreter::get().s_python_empty_tuple);

    if (!res) throw std::runtime_error("Call to tight_layout() failed.");

    Py_DECREF(res);
}

// Support for variadic plot() and initializer lists:

namespace detail {

template<typename T>
using is_function = typename std::is_function<std::remove_pointer<std::remove_reference<T>>>::type;

template<bool obj, typename T>
struct is_callable_impl;

template<typename T>
struct is_callable_impl<false, T>
{
    typedef is_function<T> type;
}; // a non-object is callable iff it is a function

template<typename T>
struct is_callable_impl<true, T>
{
    struct Fallback { void operator()(); };
    struct Derived : T, Fallback { };

    template<typename U, U> struct Check;

    template<typename U>
    static std::true_type test( ... ); // use a variadic function to make sure (1) it accepts everything and (2) its always the worst match

    template<typename U>
    static std::false_type test( Check<void(Fallback::*)(), &U::operator()>* );

public:
    typedef decltype(test<Derived>(nullptr)) type;
    typedef decltype(&Fallback::operator()) dtype;
    static constexpr bool value = type::value;
}; // an object is callable iff it defines operator()

template<typename T>
struct is_callable
{
    // dispatch to is_callable_impl<true, T> or is_callable_impl<false, T> depending on whether T is of class type or not
    typedef typename is_callable_impl<std::is_class<T>::value, T>::type type;
};

template<typename IsYDataCallable>
struct plot_impl { };

template<>
struct plot_impl<std::false_type>
{
    template<typename IterableX, typename IterableY>
    bool operator()(const IterableX& x, const IterableY& y, const std::string& format)
    {
        // 2-phase lookup for distance, begin, end
        using std::distance;
        using std::begin;
        using std::end;

        auto xs = distance(begin(x), end(x));
        auto ys = distance(begin(y), end(y));
        assert(xs == ys && "x and y data must have the same number of elements!");

        PyObject* xlist = PyList_New(xs);
        PyObject* ylist = PyList_New(ys);
        PyObject* pystring = PyString_FromString(format.c_str());

        auto itx = begin(x), ity = begin(y);
        for(size_t i = 0; i < xs; ++i) {
            PyList_SetItem(xlist, i, PyFloat_FromDouble(*itx++));
            PyList_SetItem(ylist, i, PyFloat_FromDouble(*ity++));
        }

        PyObject* plot_args = PyTuple_New(3);
        PyTuple_SetItem(plot_args, 0, xlist);
        PyTuple_SetItem(plot_args, 1, ylist);
        PyTuple_SetItem(plot_args, 2, pystring);

        PyObject* res = PyObject_CallObject(detail::Interpreter::get().s_python_function_plot, plot_args);

        Py_DECREF(plot_args);
        if(res) Py_DECREF(res);

        return res;
    }
};

template<>
struct plot_impl<std::true_type>
{
    template<typename Iterable, typename Callable>
    bool operator()(const Iterable& ticks, const Callable& f, const std::string& format)
    {
        if(begin(ticks) == end(ticks)) return true;

        // We could use additional meta-programming to deduce the correct element type of y,
        // but all values have to be convertible to double anyways
        std::vector<double> y;
        for(auto x : ticks) y.push_back(f(x));
        return plot_impl<std::false_type>()(ticks,y,format);
    }
};

} // end namespace detail

// recursion stop for the above
template<typename... Args>
bool plot() { return true; }

template<typename A, typename B, typename... Args>
bool plot(const A& a, const B& b, const std::string& format, Args... args)
{
    return detail::plot_impl<typename detail::is_callable<B>::type>()(a,b,format) && plot(args...);
}

/*
 * This group of plot() functions is needed to support initializer lists, i.e. calling
 *    plot( {1,2,3,4} )
 */
inline bool plot(const std::vector<double>& x, const std::vector<double>& y, const std::string& format = "") {
    return plot<double,double>(x,y,format);
}

inline bool plot(const std::vector<double>& y, const std::string& format = "") {
    return plot<double>(y,format);
}

inline bool plot(const std::vector<double>& x, const std::vector<double>& y, const std::map<std::string, std::string>& keywords) {
    return plot<double>(x,y,keywords);
}

/*
 * This class allows dynamic plots, ie changing the plotted data without clearing and re-plotting
 */

class Plot
{
public:
    // default initialization with plot label, some data and format
    template<typename Numeric>
    Plot(const std::string& name, const std::vector<Numeric>& x, const std::vector<Numeric>& y, const std::string& format = "") {

        assert(x.size() == y.size());

        PyObject* kwargs = PyDict_New();
        if(name != "")
            PyDict_SetItemString(kwargs, "label", PyString_FromString(name.c_str()));

        PyObject* xarray = get_array(x);
        PyObject* yarray = get_array(y);

        PyObject* pystring = PyString_FromString(format.c_str());

        PyObject* plot_args = PyTuple_New(3);
        PyTuple_SetItem(plot_args, 0, xarray);
        PyTuple_SetItem(plot_args, 1, yarray);
        PyTuple_SetItem(plot_args, 2, pystring);

        PyObject* res = PyObject_Call(detail::Interpreter::get().s_python_function_plot, plot_args, kwargs);

        Py_DECREF(kwargs);
        Py_DECREF(plot_args);

        if(res)
        {
            line= PyList_GetItem(res, 0);

            if(line)
                set_data_fct = PyObject_GetAttrString(line,"set_data");
            else
                Py_DECREF(line);
            Py_DECREF(res);
        }
    }

    // shorter initialization with name or format only
    // basically calls line, = plot([], [])
    Plot(const std::string& name = "", const std::string& format = "")
        : Plot(name, std::vector<double>(), std::vector<double>(), format) {}

    template<typename Numeric>
    bool update(const std::vector<Numeric>& x, const std::vector<Numeric>& y) {
        assert(x.size() == y.size());
        if(set_data_fct)
        {
            PyObject* xarray = get_array(x);
            PyObject* yarray = get_array(y);

            PyObject* plot_args = PyTuple_New(2);
            PyTuple_SetItem(plot_args, 0, xarray);
            PyTuple_SetItem(plot_args, 1, yarray);

            PyObject* res = PyObject_CallObject(set_data_fct, plot_args);
            if (res) Py_DECREF(res);
            return res;
        }
        return false;
    }

    // clears the plot but keep it available
    bool clear() {
        return update(std::vector<double>(), std::vector<double>());
    }

    // definitely remove this line
    void remove() {
        if(line)
        {
            auto remove_fct = PyObject_GetAttrString(line,"remove");
            PyObject* args = PyTuple_New(0);
            PyObject* res = PyObject_CallObject(remove_fct, args);
            if (res) Py_DECREF(res);
        }
        decref();
    }

    ~Plot() {
        decref();
    }
private:

    void decref() {
        if(line)
            Py_DECREF(line);
        if(set_data_fct)
            Py_DECREF(set_data_fct);
    }


    PyObject* line = nullptr;
    PyObject* set_data_fct = nullptr;
};

} // end namespace matplotlibcpp
} // end namespace MatPlotLib
} // end namespace Visualization
} // end namespace TBTK
/// @endcond
