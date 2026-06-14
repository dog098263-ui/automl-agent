package com.automl;

import java.sql.*;
import java.util.ArrayList;
import java.util.List;

public class DatabaseUtil {
    private static final String SERVER_URL = "jdbc:mysql://localhost:3306/?useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=UTC";
    private static final String URL = "jdbc:mysql://localhost:3306/datacleaning?useSSL=false&allowPublicKeyRetrieval=true&serverTimezone=UTC";
    private static final String USER = "root";
    private static final String PASSWORD = "";

    private static boolean isInitialized = false;

    static {
        initializeDatabase();
    }

    private static synchronized void initializeDatabase() {
        if (isInitialized) return;
        try {
            Class.forName("com.mysql.cj.jdbc.Driver");
        } catch (ClassNotFoundException e) {
            System.err.println("MySQL Driver class not found in DatabaseUtil: " + e.getMessage());
            return;
        }

        try (Connection conn = DriverManager.getConnection(SERVER_URL, USER, PASSWORD);
             Statement stmt = conn.createStatement()) {
            stmt.executeUpdate("CREATE DATABASE IF NOT EXISTS datacleaning;");
        } catch (SQLException e) {
            System.err.println("Failed to create datacleaning database: " + e.getMessage());
        }

        try (Connection conn = DriverManager.getConnection(URL, USER, PASSWORD);
             Statement stmt = conn.createStatement()) {
            stmt.executeUpdate(
                "CREATE TABLE IF NOT EXISTS `students_raw` (" +
                "    id INT AUTO_INCREMENT PRIMARY KEY," +
                "    `first_name` TEXT," +
                "    `last_name` TEXT," +
                "    `email` TEXT," +
                "    `phone` TEXT," +
                "    `dob` TEXT," +
                "    `gender` TEXT," +
                "    `course` TEXT," +
                "    `created_at` TEXT" +
                ");"
            );

            ResultSet rs = stmt.executeQuery("SELECT COUNT(*) FROM `students_raw`;");
            if (rs.next() && rs.getInt(1) == 0) {
                seedDirtyData(conn);
            }
            isInitialized = true;
        } catch (SQLException e) {
            System.err.println("Failed to initialize database: " + e.getMessage());
        }
    }

    private static void seedDirtyData(Connection conn) throws SQLException {
        String sql = "INSERT INTO `students_raw` (`first_name`, `last_name`, `email`, `phone`, `dob`, `gender`, `course`, `created_at`) VALUES (?, ?, ?, ?, ?, ?, ?, ?);";
        try (PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setString(1, "Aria");
            pstmt.setString(2, "Chen");
            pstmt.setString(3, "aria.chen@university.edu");
            pstmt.setString(4, "555-0192");
            pstmt.setString(5, "2004-03-12");
            pstmt.setString(6, "Female");
            pstmt.setString(7, "Computer Science");
            pstmt.setString(8, "2026-06-02 08:30:00");
            pstmt.executeUpdate();

            pstmt.setString(1, "Aria");
            pstmt.setString(2, "Chen");
            pstmt.setString(3, "aria.chen@university.edu");
            pstmt.setString(4, "555-0192");
            pstmt.setString(5, "2004-03-12");
            pstmt.setString(6, "Female");
            pstmt.setString(7, "Computer Science");
            pstmt.setString(8, "2026-06-02 08:30:00");
            pstmt.executeUpdate();

            pstmt.setString(1, "  Liam  ");
            pstmt.setString(2, "O'Connor ");
            pstmt.setString(3, " liam.oc@university.edu");
            pstmt.setString(4, "555-0143");
            pstmt.setString(5, "2003-08-22");
            pstmt.setString(6, "Male");
            pstmt.setString(7, "  Data Science ");
            pstmt.setString(8, "2026-06-02 08:32:00");
            pstmt.executeUpdate();

            pstmt.setString(1, "Sophia");
            pstmt.setString(2, "Rodriguez");
            pstmt.setString(3, "sophia.r@university.edu");
            pstmt.setString(4, "555-0188");
            pstmt.setString(5, "2004-11-05");
            pstmt.setString(6, "FEMALE");
            pstmt.setString(7, "SOFTWARE ENGINEERING");
            pstmt.setString(8, "2026-06-02 08:35:00");
            pstmt.executeUpdate();

            pstmt.setString(1, "Ethan");
            pstmt.setString(2, "Jackson");
            pstmt.setString(3, "ethan.j@university.edu");
            pstmt.setString(4, "");
            pstmt.setString(5, "2002-05-19");
            pstmt.setString(6, "Male");
            pstmt.setString(7, "Artificial Intelligence");
            pstmt.setString(8, "2026-06-02 08:40:00");
            pstmt.executeUpdate();

            pstmt.setString(1, "Mateo");
            pstmt.setString(2, "Silva");
            pstmt.setString(3, "mateo.silva@university.edu");
            pstmt.setString(4, null);
            pstmt.setString(5, "2003-09-14");
            pstmt.setString(6, "Male");
            pstmt.setString(7, null);
            pstmt.setString(8, "2026-06-02 08:42:00");
            pstmt.executeUpdate();
        }
    }

    public static Connection getConnection() throws SQLException {
        initializeDatabase();
        return DriverManager.getConnection(URL, USER, PASSWORD);
    }
}
