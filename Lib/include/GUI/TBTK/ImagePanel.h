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

/** @package TBTKcalc
 *  @file ImagePanel.h
 *  @brief Panel for displaying images.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_IMAGE_PANEL
#define COM_DAFER45_TBTK_IMAGE_PANEL

#include <wx/wx.h>
#include <wx/sizer.h>

#include <opencv2/core/core.hpp>

namespace TBTK{

class ImagePanel : public wxPanel{
public:
	/** Constructor. */
	ImagePanel(wxWindow *parent);

	/** Destructor. */
	~ImagePanel();

	/** Set image. */
	void setImage(wxString file, wxBitmapType format);

	/** Set image. */
	void setImage(const cv::Mat &image);

	/** On paint event. */
	void onPaintEvent(wxPaintEvent &event);

	/** On size event. */
	void onSizeEvent(wxSizeEvent &event);

	/** Paint. */
	void paintNow();

	/** Render. */
	void render(wxDC &dc);
protected:
	DECLARE_EVENT_TABLE();
private:
	/** The image. */
	wxImage image;

	/** Bitmap. */
	wxBitmap resizedImage;

	/** Width. */
	int width;

	/** Height. */
	int height;
};

inline void ImagePanel::setImage(const cv::Mat &image){
	unsigned char *data = new unsigned char[image.elemSize()*image.cols*image.rows];
	for(unsigned int n = 0; n < image.elemSize()*image.cols*image.rows; n++)
		data[n] = image.data[n];

	this->image = wxImage(image.cols, image.rows, data, false);
	width = -1;
	height = -1;
}

};	//End namespace TBTK

#endif
