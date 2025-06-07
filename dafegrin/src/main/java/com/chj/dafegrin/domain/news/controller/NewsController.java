package com.chj.dafegrin.domain.news.controller;

import com.chj.dafegrin.domain.news.repository.NewsRepository;
import com.chj.dafegrin.entity.News;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("/dfg/news")
public class NewsController {

    private final NewsRepository newsRepository;

    @GetMapping("/{id}")
    public ResponseEntity<?> findNewsListByDailyIndexId(@PathVariable Long id) {
        List<News> results = newsRepository.findAllByDailyIndex_Id(id);
        for(News n : results) {
            log.info(n.getContent());
        }

        return new ResponseEntity<>(results, HttpStatus.OK);
    }
}
