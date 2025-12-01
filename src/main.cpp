#include <SFML/Graphics.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <boost/asio.hpp>

cv::Mat sfml2opencv(const sf::Image &img)
{
    // get size from image
    cv::Size size(img.getSize().x, img.getSize().y);
    // create a mat on image memory
    cv::Mat mat(size, CV_8UC4, (void *)img.getPixelsPtr(), cv::Mat::AUTO_STEP);
    // make SFML RGBA to OpenCV BGRA
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    return mat.clone();
}

void sendMat(boost::asio::ip::tcp::socket &socket, const cv::Mat &mat)
{
    // Meta info
    int32_t rows = mat.rows;
    int32_t cols = mat.cols;
    int32_t type = mat.type();

    boost::asio::write(socket, boost::asio::buffer(&rows, sizeof(rows)));
    boost::asio::write(socket, boost::asio::buffer(&cols, sizeof(cols)));
    boost::asio::write(socket, boost::asio::buffer(&type, sizeof(type)));

    size_t dataSize = mat.total() * mat.elemSize();
    boost::asio::write(socket, boost::asio::buffer(mat.data, dataSize));
}

int main()
{

    boost::asio::io_context io_context;
    boost::asio::ip::tcp::acceptor acceptor(io_context,
    boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 12345));

    std::cout << "Waiting for client on port 12345..." << std::endl;

    boost::asio::ip::tcp::socket socket(io_context);
    acceptor.accept(socket);

    // drones view res
    u_int16_t drone_cam_res_x = 960;
    u_int16_t drone_cam_res_y = 540;

    u_int16_t window_res_x = 1280;
    u_int16_t window_res_y = 720;
    u_int16_t view_res_x = 1920;
    u_int16_t view_res_y = 1080;

    // square
    float rect_h = 30.f, rect_w = 20.f;
    sf::RectangleShape shape({rect_w, rect_h});
    shape.setFillColor(sf::Color::Red);

    // centering its control point
    shape.setOrigin({rect_w / 2, rect_h / 2});
    // centering it on the window
    shape.setPosition({window_res_x / 2, window_res_y / 2});

    // getting the image and creating the texture and the sprites
    sf::FileInputStream bg_stream(std::string(ASSETS_DIR) + "/dag.jpg");
    sf::Texture bg_texture(bg_stream);
    sf::Sprite bg_sprite(bg_texture);

    // creating the window and setting vsync so my pc doesnt die
    sf::RenderWindow window(sf::VideoMode({window_res_x, window_res_y}), "!", sf::Style::None, sf::State::Windowed);
    window.setVerticalSyncEnabled(true); // call it once after creating the window

    // setting the sprite's starting point to texture's 0,0 and its size to the window size
    // bg_sprite.setTextureRect(sf::IntRect({0, 0}, {window_res_x, window_res_y}));

    // speed of the little dot
    auto speed_x = 3.5f, speed_y = 3.5f;
    // space changed by the little dot
    auto tomove_x = 0.0f, tomove_y = 0.0f;
    // offset values for checks
    auto offset_x = 2.0f, offset_y = 2.0f;

    // second window for debug
    sf::RenderWindow window2(sf::VideoMode({drone_cam_res_x, drone_cam_res_y}), "SFML works!", sf::Style::None, sf::State::Windowed);
    window2.setVerticalSyncEnabled(true); // call it once after creating the window

    sf::Vector2f view_pos(shape.getPosition().x, shape.getPosition().y);
    while (window.isOpen())
    {
        // bg_sprite.setTextureRect(sf::IntRect({shape.getPosition().x, shape.getPosition().y}, {window_res_x, window_res_y}));

        // std::cout << std::string(ASSETS_DIR) + "/dag.jpg";

        // HANDLE KEYBOARD EVENTS
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Left))
        {
            // std::cout << "move left";
            // tomove_x = -speed_x;
            shape.rotate(sf::degrees(-2.5f));
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Right))
        {
            // std::cout << "move right ";
            // tomove_x = speed_x;
            shape.rotate(sf::degrees(2.5f));
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Down))
        {
            // std::cout << "move left";
            tomove_y = speed_y;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Up))
        {
            // std::cout << "move right ";
            tomove_y = -speed_y;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space))
        {
            // std::cout << "move right ";
            speed_y += 2.5f;
            speed_x += 2.5f;
            offset_x = speed_x * 2, offset_y = speed_y * 2;
        }

        // HANDLE LITTLE DOTS MOVEMENT

        float rad = shape.getRotation().asDegrees() * 3.14159265f / 180.f;
        sf::Vector2f moveVec(std::sin(rad) * tomove_y, -std::cos(rad) * tomove_y);

        // Calculate future position
        sf::Vector2f futurePos = shape.getPosition() + moveVec;

        // Get background texture size
        sf::Vector2u texSize2 = bg_texture.getSize();

        // Optionally, consider shape size to keep entire shape inside bounds
        sf::Vector2f shapeSize = shape.getSize();
        sf::Vector2f halfShapeSize = shapeSize / 2.f;

        // Clamp future position so shape stays fully inside texture bounds
        float minX = halfShapeSize.x;
        float maxX = texSize2.x - halfShapeSize.x;
        float minY = halfShapeSize.y;
        float maxY = texSize2.y - halfShapeSize.y;

        if (futurePos.x >= minX && futurePos.x <= maxX &&
            futurePos.y >= minY && futurePos.y <= maxY)
        {
            shape.move(moveVec);
        }

        tomove_x = 0.0f;
        tomove_y = 0.0f;

        std::cout << shape.getPosition().x << " " << shape.getPosition().y << std::endl;
        // std::cout << shape.getPosition().x << " " << shape.getPosition().y << std::endl;

        // std::cout << max_window_x << " " << max_window_y << " " <<curr_window_x << " " << curr_window_y << std::endl;
        sf::View view1;

        sf::Vector2u texSize = bg_texture.getSize();

        // Half sizes of the view (window)
        float halfViewW = view_res_x / 2.f;
        float halfViewH = view_res_y / 2.f;

        // Get the shape position
        sf::Vector2f shapePos = shape.getPosition();

        // Clamp the view position so it stays within texture bounds
        sf::Vector2f view_pos = shapePos;

        // Clamp X
        if (view_pos.x < halfViewW)
            view_pos.x = halfViewW;
        else if (view_pos.x > texSize.x - halfViewW)
            view_pos.x = texSize.x - halfViewW;

        // Clamp Y
        if (view_pos.y < halfViewH)
            view_pos.y = halfViewH;
        else if (view_pos.y > texSize.y - halfViewH)
            view_pos.y = texSize.y - halfViewH;

        view1.setCenter(view_pos);
        view1.setSize({view_res_x, view_res_y});
        window.setView(view1);
        window.draw(bg_sprite);
        window.draw(shape);

        window.display();

        sf::View view;
        view.setCenter(shape.getPosition());
        view.setSize({drone_cam_res_x, drone_cam_res_y});
        view.setRotation(shape.getRotation());
        window2.setView(view);
        window2.clear();
        window2.draw(bg_sprite);
        // window2.draw(shape);
        // window2.display();

        // get the image
        sf::Texture windowTexture({window2.getSize().x, window2.getSize().y});
        windowTexture.update(window2);
        sf::Image croppedImage = windowTexture.copyToImage();
        // cv::imshow("drone_view", sfml2opencv(croppedImage));
        // cv::waitKey(1);
        sendMat(socket, sfml2opencv(croppedImage));
        float angle = shape.getRotation().asDegrees() ;
        boost::asio::write(socket, boost::asio::buffer(&angle, sizeof(angle)));
    }
}
