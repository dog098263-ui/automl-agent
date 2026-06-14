package com.automl;

import java.io.*;
import java.sql.*;
import java.util.*;
import jakarta.servlet.http.*;
import jakarta.servlet.ServletException;
import com.google.gson.Gson;

public class AutoMLServlet extends HttpServlet {
    private final Gson gson = new Gson();

    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        String path = req.getPathInfo();
        if (path == null) path = "/";
        
        resp.setContentType("application/json");
        PrintWriter writer = resp.getWriter();
        
        if (path.equals("/health")) {
            writer.print(gson.toJson(Map.of("status", "ok", "service", "Java Tomcat Servlet")));
        } else {
            resp.sendError(404);
        }
    }

    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        String path = req.getPathInfo();
        if (path == null) path = "/";
        
        resp.setContentType("application/json");
        resp.setHeader("Access-Control-Allow-Origin", "*");
        resp.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
        resp.setHeader("Access-Control-Allow-Headers", "Content-Type");
        
        PrintWriter writer = resp.getWriter();
        
        if (path.equals("/db/load")) {
            // Load table from MySQL and return JSON rows & columns
            try {
                Map params = gson.fromJson(req.getReader(), Map.class);
                String table = (String) params.get("table");
                if (table == null || table.trim().isEmpty()) {
                    resp.setStatus(400);
                    writer.print(gson.toJson(Map.of("error", "Table name is required")));
                    return;
                }
                
                Map<String, Object> data = loadTable(table);
                writer.print(gson.toJson(data));
            } catch (Exception e) {
                resp.setStatus(500);
                writer.print(gson.toJson(Map.of("error", e.getMessage())));
            }
        } else if (path.equals("/db/save")) {
            // Save table back to MySQL
            try {
                Map params = gson.fromJson(req.getReader(), Map.class);
                String table = (String) params.get("table");
                List<String> columns = (List<String>) params.get("columns");
                List<List<String>> rows = (List<List<String>>) params.get("rows");
                
                if (table == null || columns == null || rows == null) {
                    resp.setStatus(400);
                    writer.print(gson.toJson(Map.of("error", "Missing table, columns, or rows")));
                    return;
                }
                
                saveTable(table, columns, rows);
                writer.print(gson.toJson(Map.of("message", "Successfully saved to MySQL table " + table)));
            } catch (Exception e) {
                resp.setStatus(500);
                writer.print(gson.toJson(Map.of("error", e.getMessage())));
            }
        } else {
            resp.sendError(404);
        }
    }

    @Override
    protected void doOptions(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        resp.setHeader("Access-Control-Allow-Origin", "*");
        resp.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
        resp.setHeader("Access-Control-Allow-Headers", "Content-Type");
        resp.setStatus(200);
    }

    private Map<String, Object> loadTable(String table) throws SQLException {
        String safeTable = table.replaceAll("`", "");
        String sql = "SELECT * FROM `" + safeTable + "`;";
        
        try (Connection conn = DatabaseUtil.getConnection();
             Statement stmt = conn.createStatement();
             ResultSet rs = stmt.executeQuery(sql)) {
            
            ResultSetMetaData meta = rs.getMetaData();
            int colCount = meta.getColumnCount();
            
            List<String> cols = new ArrayList<>();
            for (int i = 1; i <= colCount; i++) {
                String colName = meta.getColumnName(i);
                if (colName.equalsIgnoreCase("id") && colCount > 1) {
                    continue;
                }
                cols.add(colName);
            }
            
            List<Map<String, String>> dataRows = new ArrayList<>();
            List<List<String>> rawRows = new ArrayList<>();
            while (rs.next()) {
                List<String> rawRow = new ArrayList<>();
                Map<String, String> rowMap = new LinkedHashMap<>();
                for (int i = 1; i <= colCount; i++) {
                    String colName = meta.getColumnName(i);
                    if (colName.equalsIgnoreCase("id") && colCount > 1) {
                        continue;
                    }
                    String val = rs.getString(i);
                    rawRow.add(val == null ? "" : val);
                    rowMap.put(colName, val == null ? "" : val);
                }
                rawRows.add(rawRow);
                dataRows.add(rowMap);
            }
            
            Map<String, Object> result = new HashMap<>();
            result.put("columns", cols);
            result.put("rows", dataRows.size());
            result.put("data", dataRows);
            result.put("raw_rows", rawRows);
            return result;
        }
    }

    private void saveTable(String table, List<String> columns, List<List<String>> rows) throws SQLException {
        String safeTable = table.replaceAll("`", "");
        
        try (Connection conn = DatabaseUtil.getConnection()) {
            StringBuilder createSql = new StringBuilder("CREATE TABLE IF NOT EXISTS `")
                    .append(safeTable)
                    .append("` (id INT AUTO_INCREMENT PRIMARY KEY, ");
            
            for (int i = 0; i < columns.size(); i++) {
                createSql.append("`").append(columns.get(i)).append("` TEXT");
                if (i < columns.size() - 1) {
                    createSql.append(", ");
                }
            }
            createSql.append(");");
            
            try (Statement stmt = conn.createStatement()) {
                stmt.execute(createSql.toString());
            }

            StringBuilder insertSql = new StringBuilder("INSERT INTO `")
                    .append(safeTable)
                    .append("` (");
            for (int i = 0; i < columns.size(); i++) {
                insertSql.append("`").append(columns.get(i)).append("`");
                if (i < columns.size() - 1) {
                    insertSql.append(", ");
                }
            }
            insertSql.append(") VALUES (");
            for (int i = 0; i < columns.size(); i++) {
                insertSql.append("?");
                if (i < columns.size() - 1) {
                    insertSql.append(", ");
                }
            }
            insertSql.append(");");

            try (PreparedStatement pstmt = conn.prepareStatement(insertSql.toString())) {
                for (List<String> row : rows) {
                    for (int i = 0; i < columns.size(); i++) {
                        String val = i < row.size() ? row.get(i) : null;
                        pstmt.setString(i + 1, val);
                    }
                    pstmt.executeUpdate();
                }
            }
        }
    }
}
