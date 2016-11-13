#ifndef COM_DAFER45_TBTK_VISUALIZER_MAIN_APPLICATION
#define COM_DAFER45_TBTK_VISUALIZER_MAIN_APPLICATION

#include <OgreWindowEventUtilities.h>
#include <OgreFrameListener.h>
#include <OgreRoot.h>
#include <OgreRenderWindow.h>
#include <OgreSceneManager.h>
#include <OgreCamera.h>
#include <OgreViewport.h>
#include <OISEvents.h>
#include <OISInputManager.h>
#include <OISKeyboard.h>
#include <OISMouse.h>

#include <string>

class MainApplication : public Ogre::WindowEventListener, public Ogre::FrameListener, public OIS::KeyListener, public OIS::MouseListener
{
public:
	MainApplication();
	virtual ~MainApplication();

	void setFileName(std::string fileName);

	bool go();
protected:
	Ogre::Root *mRoot;
	Ogre::String mResourcesCfg;
	Ogre::String mPluginsCfg;
	Ogre::String mConfigCfg;
	Ogre::String mLog;

	Ogre::RenderWindow *mWindow;
	Ogre::SceneManager *mSceneManager;
	Ogre::Camera *mCamera;
	Ogre::Viewport *mViewport;

	Ogre::Real mRotate;
	Ogre::Real mMove;
	Ogre::SceneNode *cameraNode;
	Ogre::Vector3 cameraDirection;
	Ogre::Light *light;

	OIS::InputManager *mInputManager;
	OIS::Keyboard *mKeyboard;
	OIS::Mouse *mMouse;

	bool mShutDown;

	virtual void createScene();

	//Overrides WindowEventListener
	virtual void windowResized(Ogre::RenderWindow *rw);
	virtual void windowClosed(Ogre::RenderWindow *rw);

	//Overrides KeyListener
	virtual bool keyPressed(const OIS::KeyEvent &ke);
	virtual bool keyReleased(const OIS::KeyEvent &ke);

	//Overrides MouseListener
	virtual bool mouseMoved(const OIS::MouseEvent &me);
	virtual bool mousePressed(const OIS::MouseEvent &me, OIS::MouseButtonID id);
	virtual bool mouseReleased(const OIS::MouseEvent &me, OIS::MouseButtonID id);

	//Overrider FrameListener
	virtual bool frameRenderingQueued(const Ogre::FrameEvent &event);

	int screenshotCounter;
private:
	std::string fileName;
	Ogre::SceneNode *geometryNode;

	bool init();
	void loadGeometry();
	void deleteGeometry();
	void destroyAllAttachedMovableObjects(Ogre::SceneNode *node);
};

inline void MainApplication::setFileName(std::string fileName){
	this->fileName = fileName;
}

#endif
