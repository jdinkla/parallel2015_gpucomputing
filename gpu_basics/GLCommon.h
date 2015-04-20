/*
* Copyright (c) 2015 by Joern Dinkla, www.dinkla.com, All rights reserved.
*
* See the LICENSE file in the root directory.
*/

#pragma once


#include <iostream>

void cleanupGL();
void idle();
void display();
void render();
void keyboard(unsigned char key, int x, int y);
void initOpenGLBuffers(const int width, const int height);
void reshape(int width, int height);
void cleanup();

shared_ptr<IDemo> demo;					// the demo

FPS fps;
int device = 0;							// the GPU

float min_fps = 1000000.0f;
float max_fps = 0;
float sum_fps = 0;
int count_fps = 0;

int width = 1280;						// the width and height of the window
int height = 720;

bool profiling = false;					// are we profiling or demo?
int profiling_iterations_left = 0;		// how many iterations to go
bool animate = true;					// should the animation run?
bool changed = false;					// has the content changed by CUDA?

int step = 0;
int constant_step = 57;
int constant_mod = 100;
bool add_heat = true;

// TODO p15
//#ifdef WINDOWS
//const char* windowTitle = "Demo zur Parallel 2015 von J\366rn Dinkla";
//const char* windowTitle2 = "Demo zur Parallel 2015 von J\366rn Dinkla [%3.1f fps]";
//#else
//const char* windowTitle = "Demo zur Parallel 2015 von Joern Dinkla";
//const char* windowTitle2 = "Demo zur Parallel 2015 von Joern Dinkla [%3.1f fps]";
//#endif

#ifdef WINDOWS
const char* windowTitle = "Heat diffusion demo by J\366rn Dinkla";
const char* windowTitle2 = "Heat diffusion demo by J\366rn Dinkla [%3.1f fps]";
#else
const char* windowTitle = "Heat diffusion demo by Joern Dinkla";
const char* windowTitle2 = "Heat diffusion demo by Joern Dinkla [%3.1f fps]";
#endif

GLuint pbo = 0;										// OpenGL pixel buffer object
void* d_image = nullptr; 							// image on the device

#ifdef WINDOWS
// This is specifically to enable the application to enable/disable vsync
typedef BOOL(WINAPI *PFNWGLSWAPINTERVALFARPROC)(int);

void setVSync(int interval)
{
	if (WGL_EXT_swap_control)
	{
		wglSwapIntervalEXT = (PFNWGLSWAPINTERVALFARPROC)wglGetProcAddress("wglSwapIntervalEXT");
		wglSwapIntervalEXT(interval);
	}
}
#endif

inline void reset_fps()
{
	min_fps = 1000000.0f;
	max_fps = 0.0f;
	count_fps = 0;
	sum_fps = 0.0f;
}

inline void print_data(const char* name)
{
	glDisable(GL_LIGHTING);
	glColor3f(1, 1, 1);
	glRasterPos2f(0.01f, 0.01f);
	stringstream str2;
	str2 << name << ", " << width << " x " << height << ", FPS: min: " << min_fps << ", max: " << max_fps << ", avg: " << (sum_fps / count_fps);
	for (auto c : str2.str())
	{
		glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, int{ c });
	}

	stringstream str;
	str << ", step: " << step << (add_heat ? ", adding heat" : "");
	for (auto c : str.str())
	{
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, int(c));
	}
}

// Compute the FPS and set the window title
void computeFPS()
{
	Maybe<float> m = fps.update();
	if (m.is_just())
	{
		const float new_fps = m.get_value();
		sum_fps += new_fps;
		count_fps++;
		if (new_fps < min_fps) min_fps = new_fps;
		if (new_fps > max_fps) max_fps = new_fps;
		char buf[256];
		sprintf(buf, windowTitle2, new_fps);
		glutSetWindowTitle(buf);

		stringstream str2;
		str2 << width << " x " << height << ", FPS: min: " << min_fps << ", max: " << max_fps << ", avg: " << (sum_fps / count_fps);
		cout << str2.str() << endl;
	}
}

// initialize GLUT callback functions
void initGL(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow(windowTitle);
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);
	glutIdleFunc(idle);
	glutCloseFunc(cleanup);

	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"))
	{
		cerr << "Required OpenGL extensions missing." << endl;
		exit(-1);
	}

#ifdef MAC_TODO // TODO this does not work any more
	// We need unlimited FPS
	GLint p = 0;
	CGLError rc = CGLSetParameter(CGLGetCurrentContext(), kCGLCPSwapInterval, &p);
#endif

}


void keyboard(unsigned char key, int x, int y)
{
	if (profiling)			// not if profiling
	{
		return;
	}
	switch (key)
	{
	case 'r':
	case 'R':
		reshape(width, height);
		break;
	case 's':
	case 'S':
	{
		stringstream filename;
		filename << "demo." << step << ".bmp";
		demo->save(filename.str());
	}
	break;
	case 27:
	case 'Q':
	case 'q':
		exit(0);
		break;
	case ' ':
		animate = !animate;
		break;
	case 'h':
		demo->set_heat_source(ConstantHeatSource::make_shared(constant_step, constant_mod));
		demo->cleanup();
		demo->init(width, height);
		break;
	case 'H':
		demo->set_heat_source(RandomHeatSource::make_shared());
		demo->cleanup();
		demo->init(width, height);
		break;
	case 'o':
		if (constant_step > 1)
		{
			constant_step--;
			cout << "constant_step=" << constant_step << ", constant_mod=" << constant_mod << endl;
			demo->set_heat_source(ConstantHeatSource::make_shared(constant_step, constant_mod));
			demo->cleanup();
			demo->init(width, height);
		}
		break;
	case 'p':
		constant_step++;
		cout << "constant_step=" << constant_step << ", constant_mod=" << constant_mod << endl;
		demo->set_heat_source(ConstantHeatSource::make_shared(constant_step, constant_mod));
		demo->cleanup();
		demo->init(width, height);
		break;
	case 'O':
		if (constant_mod > 1)
		{
			constant_mod--;
			cout << "constant_step=" << constant_step << ", constant_mod=" << constant_mod << endl;
			demo->set_heat_source(ConstantHeatSource::make_shared(constant_step, constant_mod));
			demo->cleanup();
			demo->init(width, height);
		}
		break;
	case 'P':
		constant_mod++;
		cout << "constant_step=" << constant_step << ", constant_mod=" << constant_mod << endl;
		demo->set_heat_source(ConstantHeatSource::make_shared(constant_step, constant_mod));
		demo->cleanup();
		demo->init(width, height);
		break;
	case 'a':
	case 'A':
		add_heat = !add_heat;
		cout << "setting add_heat to " << add_heat << endl;
		demo->set_add_heat(add_heat);
		break;
	case '1':
		glutReshapeWindow(1280, 720);
		break;
	case '2':
		glutReshapeWindow(1920, 1080);
		break;
	case '3':
		glutReshapeWindow(2560, 1440);
		break;
	case '4':
		glutPositionWindow(0, 0);
		glutReshapeWindow(3840, 2160);
		break;
	default:
		break;
	}
}

void cleanup()
{
	fps.stop_timer();
	fps.delete_timer();
	cleanupGL();
	demo->cleanup();
	if (d_image)
	{
		free(d_image);
		d_image = nullptr;
	}
}

