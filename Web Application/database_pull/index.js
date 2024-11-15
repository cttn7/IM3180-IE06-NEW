// index.js
const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const path = require('path');
const app = express();
const port = 3000;
// display excel 
const ExcelJS = require('exceljs');

app.use(cors());
app.use(express.json());

// Serve static files from the "public" directory
app.use(express.static(path.join(__dirname, 'public')));

// Create a MySQL connection
const db = mysql.createConnection({
    host: 'localhost',
    user: 'root', // Change to your MySQL username
    password: '1234', // Change to your MySQL password
    database: 'data_demo1' // Change to your database name
});

// Connect to the database
db.connect(err => {
    if (err) {
        console.error('Database connection failed:', err);
        return;
    }
    console.log('Connected to the database.');
});

// Endpoint to get posts with optional category filter
app.get('/posts', (req, res) => {
    const category = req.query.category;
    let sql = 'SELECT * FROM posts';

    if (category) {
        sql += ' WHERE category = ?';
        db.query(sql, [category], (err, results) => {
            if (err) {
                return res.status(500).send(err);
            }
            res.json(results);
        });
    } else {
        db.query(sql, (err, results) => {
            if (err) {
                return res.status(500).send(err);
            }
            res.json(results);
        });
    }
});

// API endpoint to search for posts by title
app.get('/search', (req, res) => {
    const searchTerm = req.query.query ? req.query.query.toLowerCase() : '';
    const category = req.query.category;

    if (!searchTerm) {
        return res.json([]);
    }

    let sql = 'SELECT * FROM posts WHERE LOWER(title) LIKE ?';
    const params = [`%${searchTerm}%`];

    if (category) {
        sql += ' AND category = ?';
        params.push(category);
    }

    db.query(sql, params, (err, results) => {
        if (err) {
            return res.status(500).json(err);
        }
        res.json(results);
    });
});

app.get('/posts/:id', (req, res) => {
    const id = parseInt(req.params.id);
    console.log(`Fetching post with ID: ${id}`); // Debugging line
    db.query('SELECT * FROM posts WHERE id = ?', [id], (err, results) => {
        if (err) {
            console.error('Database error:', err); // Log database errors
            return res.status(500).send(err);
        }
        if (results.length > 0) {
            console.log('Post found:', results[0]); // Log the result
            res.json(results[0]);
        } else {
            console.log('Post not found'); // Log if the post was not found
            res.status(404).json({ error: 'Post not found' });
        }
    });
});

// get the excel sheet 
app.get('/generate-excel/:postId', (req, res) => {
    const postId = req.params.postId;
    db.query('SELECT * FROM your_table WHERE post_id = ?', [postId], async (error, results) => {
        if (error) {
            return res.status(500).json({ error: 'Database query error' });
        }

        const workbook = new ExcelJS.Workbook();
        const worksheet = workbook.addWorksheet('Sheet 1');

        worksheet.columns = [
            { header: 'Column1', key: 'column1' },
            { header: 'Column2', key: 'column2' },
            // Add more columns as needed
        ];

        results.forEach((row) => {
            worksheet.addRow(row);
        });

        res.setHeader(
            'Content-Type',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        );
        res.setHeader('Content-Disposition', 'attachment; filename=data.xlsx');

        await workbook.xlsx.write(res);
        res.end();
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
